"""Validate the proportional-rope (`rope_freqs.weight`) semantics with tiny models.

Builds two tiny mixed-layer gemma4 GGUFs (1 sliding + 1 full attention layer,
head dim 8 / 16) that differ ONLY in the `rope_freqs.weight` table:
  * ones:      every pair rotates (plain full rotary, the historical file)
  * sentinel:  25% of the pairs rotate, the rest use llama.cpp's 1e30
               no-rotation sentinel -- the layout of the real gemma4-12b file.

llama.cpp b12559 asserts gemma4 has at least one sliding layer, so the model must
stay mixed; the multi-token prompt (T=8 > window) exercises rope at positions > 0
on the full layer.  Sliding-window semantics are pinned down first with the
"ones" model (both lt/le mask conventions are tried) so the sentinel comparison
isolates the frequency-factor rule.

Expected: with the window convention llama.cpp uses, the "ones" model matches
both reference variants, and the "sentinel" model matches ONLY when the pair
angle is divided by the table (rope_ff=auto).

Usage: python tools/check_rope_ff_tiny.py [--port 8094] [--ids 3,5,7,9,11,13,4,6]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma4_ref import GGUFTensors, Config  # noqa: E402
from tiny_gemma4_grid import forward, base_variant  # noqa: E402
from make_tiny_gemma4 import build as build_tiny  # noqa: E402

LLAMA = Path("U:/llama/llama-server.exe")
# Synthetic full-layer rope base: smaller than the real 1e6 so the 1e30
# sentinel pairs produce visible angles already at 8 prompt tokens; the
# rule under test (divide the angle by the table) is base-independent and
# the same model is what llama.cpp is asked about.
PROBE_FULL_BASE = 100.0
# Amplify the full-attention layer so its output is a visible part of the
# residual stream (the real weights make it small; the rule is unchanged).
PROBE_WEIGHT_SCALE = 6.0
# Make the full layer dominate the residual stream so its rope changes are
# visible on the final hidden state (the real weights hide them behind the
# residual; the rule under test is unchanged).


def post(port: int, path: str, obj: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}" + path,
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def start_server(port: int, model: Path) -> subprocess.Popen:
    if not LLAMA.exists():
        raise SystemExit(f"llama-server not found at {LLAMA}")
    p = subprocess.Popen(
        [str(LLAMA), "-m", str(model), "--ctx-size", "128", "-ngl", "0",
         "--host", "127.0.0.1", "--port", str(port), "--temp", "0",
         "--top-k", "1", "--parallel", "1", "--embeddings"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit("llama-server died while loading the tiny model")
            time.sleep(0.5)
    p.kill()
    raise SystemExit("llama-server did not become ready")


def query_oracle(port: int, ids: list[int], n_probs: int = 10) -> dict:
    c = post(port, "/completion", {"prompt": ids, "n_predict": 2, "temperature": 0.0,
                                   "n_probs": n_probs, "cache_prompt": False})
    e = post(port, "/embedding", {"content": ids})
    hidden = np.array(e[0]["embedding"], dtype=np.float32)
    step0 = c["completion_probabilities"][0]["top_logprobs"]
    return {"hidden": torch.from_numpy(hidden).float(),
            "logp": {int(t["id"]): float(t["logprob"]) for t in step0},
            "gen": [int(t["id"]) for t in c["completion_probabilities"]]}


def reference(model: Path, ids: list[int], rope_ff: str, win_mode: str):
    ts = GGUFTensors(model)
    cfg = Config(ts)
    v = base_variant()
    v["rope_ff"] = rope_ff
    v["win_mode"] = win_mode
    return forward(ts, cfg, ids, v)


def compare(oracle: dict, hidden: torch.Tensor, logits: torch.Tensor) -> tuple[float, float, int]:
    target = oracle["hidden"][-1]
    rel = float((hidden[-1] - target).norm() / target.norm())
    lp = torch.log_softmax(logits[-1].float(), dim=-1)
    dlogp = max(abs(float(lp[i]) - p) for i, p in oracle["logp"].items())
    return rel, dlogp, int(torch.topk(lp, 1).indices[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--ids", default="3,5,7,9,11,13,4,6")
    opts = ap.parse_args()
    ids = [int(x) for x in opts.ids.split(",")]
    torch.set_num_threads(4)

    fixtures = Path("build/fixtures")
    models = {}
    for mode in ("ones", "sentinel", "none"):
        models[mode] = build_tiny(fixtures / f"tiny_gemma4_rope_{mode}", mode,
                                  full_rope_base=PROBE_FULL_BASE,
                                  full_layer_weight_scale=PROBE_WEIGHT_SCALE,
                                  probe_full_dominant=True)

    oracle = {}
    for mode, model in models.items():
        server = start_server(opts.port, model)
        try:
            oracle[mode] = query_oracle(opts.port, ids)
        finally:
            server.kill()
            server.wait()

    # ---- 1) pin the sliding-window convention with the all-ones model -------
    win_rows = []
    for win_mode in ("lt", "le"):
        hidden, logits = reference(models["ones"], ids, "auto", win_mode)
        rel, dlogp, top = compare(oracle["ones"], hidden, logits)
        win_rows.append((rel, win_mode, dlogp, top))
        print(f"ones     win_mode={win_mode:2s} relL2={rel:.3e} max|dlogp|={dlogp:.3e} "
              f"ref_top={top} llama_gen0={oracle['ones']['gen'][0]}")
    win_rows.sort()
    rel_win, win_mode, dlogp_win, top_win = win_rows[0]

    # ---- 2) sentinel model: with / without the frequency factors -----------
    sent_rows = {}
    for rope_ff in ("auto", "ignore"):
        hidden, logits = reference(models["sentinel"], ids, rope_ff, win_mode)
        rel, dlogp, top = compare(oracle["sentinel"], hidden, logits)
        sent_rows[rope_ff] = (rel, dlogp, top)
        print(f"sentinel rope_ff={rope_ff:6s} relL2={rel:.3e} max|dlogp|={dlogp:.3e} "
              f"ref_top={top} llama_gen0={oracle['sentinel']['gen'][0]}")

    none_rows = {}
    for rope_ff in ("auto", "ignore"):
        hidden, logits = reference(models["none"], ids, rope_ff, win_mode)
        rel, dlogp, top = compare(oracle["none"], hidden, logits)
        none_rows[rope_ff] = (rel, dlogp, top)
        print(f"none     rope_ff={rope_ff:6s} relL2={rel:.3e} max|dlogp|={dlogp:.3e} "
              f"ref_top={top} llama_gen0={oracle['none']['gen'][0]}")

    checks = 0
    fails = 0

    def check(name, ok):
        nonlocal checks, fails
        checks += 1
        if not ok:
            fails += 1
            print(f"  FAIL {name}")

    check("ones model matches llama.cpp (window convention pinned, relL2 < 1e-3)",
          rel_win < 1e-3)
    check("ones model log-probs match llama.cpp (< 1e-3)", dlogp_win < 1e-3)
    check("ones model top-1 matches llama.cpp", top_win == oracle["ones"]["gen"][0])

    # sentinel = the real 12b layout (64 of 256 pairs rotate; here 2 of 8).
    rel_auto, dlogp_auto, top_auto = sent_rows["auto"]
    rel_ign, _dlogp_ign, _ = sent_rows["ignore"]
    check("sentinel: rope_ff=auto matches llama hidden (relL2 < 1e-3)", rel_auto < 1e-3)
    check("sentinel: rope_ff=auto matches llama log-probs (< 1e-3)", dlogp_auto < 1e-3)
    check("sentinel: rope_ff=auto top-1 matches llama", top_auto == oracle["sentinel"]["gen"][0])
    check("sentinel: ignoring the table is measurably wrong (relL2 > 1.5e-3)", rel_ign > 1.5e-3)

    # none = degenerate proportional table (no pair rotates on the full layer):
    # the strongest sensitivity case, it proves the divide rule itself.
    rel_none_auto, dlogp_none_auto, _top_none = none_rows["auto"]
    rel_none_ign, _, _ = none_rows["ignore"]
    check("none: rope_ff=auto matches llama hidden (relL2 < 1e-3)", rel_none_auto < 1e-3)
    check("none: rope_ff=auto matches llama log-probs (< 1e-3)", dlogp_none_auto < 1e-3)
    check("none: ignoring the table is very wrong (relL2 > 3e-3)", rel_none_ign > 3e-3)

    print(f"\nROPE FREQ FACTORS vs llama.cpp: {checks} checks, {fails} failed "
          f"(window convention: {win_mode})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
