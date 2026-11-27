"""Collect the llama.cpp greedy oracle for the gemma4-12B parity run (CPU, no VRAM)."""
import json, os, subprocess, time, urllib.request
from pathlib import Path

# Paths come from the environment (CI no-private-paths: no machine-specific literals).
LLAMA = os.environ.get("LLAMA_SERVER") or "llama-server.exe"
_default_model = os.path.join(os.environ.get("MODEL_DIR", ""),
                              "gemma-4-12b-it-Q4_K_M.gguf")
MODEL = os.environ.get("GEMMA_MODEL") or (_default_model if os.environ.get("MODEL_DIR") else "")
if not MODEL:
    raise SystemExit("set MODEL_DIR (or GEMMA_MODEL) to the gemma-4-12b gguf")
PORT = 8092
PROMPTS = {
    "hello": [9259],
    "nihao": [144626],
    "france": [818, 5279, 529, 7001, 563],
}

def post(path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}" + path,
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

p = subprocess.Popen([LLAMA, "-m", MODEL, "--ctx-size", "256", "-ngl", "0",
                      "--host", "127.0.0.1", "--port", str(PORT), "--temp", "0",
                      "--top-k", "1", "--parallel", "1"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            break
        except Exception:
            if p.poll() is not None:
                raise SystemExit("llama-server died")
            time.sleep(0.5)
    out = {}
    for name, ids in PROMPTS.items():
        r = post("/completion", {"prompt": ids, "n_predict": 16, "temperature": 0.0,
                                 "n_probs": 5, "cache_prompt": False})
        steps = r["completion_probabilities"]
        out[name] = {"ids": ids,
                     "gen": [s["id"] for s in steps],
                     "top0": [(t["id"], round(t["logprob"], 5)) for t in steps[0]["top_logprobs"][:5]]}
        print(name, "gen", out[name]["gen"], flush=True)
    Path("build/parity-oracle.json").write_text(json.dumps(out, indent=1))
    print("saved build/parity-oracle.json")
finally:
    p.kill()
    p.wait()
