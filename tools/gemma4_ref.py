"""Gemma4 reference forward in torch (CPU) built from the GGUF file.

Purpose: pin down the exact architecture semantics of a *real* GGUF that this
engine had never been adapted to, before porting it into the C++/CUDA engine.
The implementation is metadata-driven (per-layer head counts / head dims /
sliding window / rope base / optional tensors) so it doubles as the spec for the
generic engine path.

It is compared against llama.cpp (llama-server HTTP API) for the same prompt:
  * greedy token ids for N steps,
  * top-k log-probabilities at the first generated position.

Usage (llama-server for the same model running on :8099):
  python tools/gemma4_ref.py --prompt "Hello" --n 8
  python tools/gemma4_ref.py --prompt "你好" --norm-mode plus1 --rope-full 0

Hypothesis switches (defaults = current best guess from the HF gemma4 source):
  --norm-mode direct|plus1     RMSNorm uses stored weight directly or (1+w)
  --qk-norm-mode direct|plus1  same for attn_q_norm/attn_k_norm
  --attn-scale 1.0|rsqrt       attention logit scale
  --rope-full 1|0              full rotary (GGUF rope.dimension_count) vs 25% partial
  --embed-scale sqrt|none      embedding * sqrt(hidden) or not
  --v-norm on|off              weightless RMSNorm on V
  --sandwich on|off            post_attention_norm / post_ffw_norm
  --layer-scalar on|off        multiply layer output by layer_output_scale
  --softcap on|off             final logit softcap
  --act gelu|silu              MLP activation
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import gguf
    from gguf.quants import dequantize
except ImportError:  # pragma: no cover
    raise SystemExit("需要 pip install gguf (llama.cpp 的 gguf-py)")

MODEL = Path("U:/AI/gemma-4-12b-it-Q4_K_M.gguf")
SERVER = "http://127.0.0.1:8099"
TOP = 10


# ---------------------------------------------------------------------------
# llama-server oracle
# ---------------------------------------------------------------------------


def post(path: str, obj: dict) -> dict:
    req = urllib.request.Request(SERVER + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))


def oracle(prompt: str, n: int) -> dict:
    ids = post("/tokenize", {"content": prompt})["tokens"]
    r = post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0.0,
                             "n_probs": TOP, "cache_prompt": False})
    probs = r.get("completion_probabilities", [])
    return {"ids": ids, "steps": probs, "gen": [p["id"] for p in probs]}


# ---------------------------------------------------------------------------
# GGUF tensor access with per-layer dequantisation
# ---------------------------------------------------------------------------


class GGUFTensors:
    def __init__(self, path: Path):
        self.reader = gguf.GGUFReader(str(path), "r")
        self.by = {t.name: t for t in self.reader.tensors}
        self.cache: dict[str, np.ndarray] = {}
        self.fields: dict[str, object] = {}
        for k, field in self.reader.fields.items():
            if field.count == 1:
                vals = field.contents()
                self.fields[k] = vals[0]
            else:
                self.fields[k] = field.contents()
        arch = self.fields["general.architecture"]
        if isinstance(arch, bytes):
            arch = arch.decode()
        self.arch = arch

    def has(self, name: str) -> bool:
        return name in self.by

    def logical_shape(self, name: str) -> tuple:
        return tuple(reversed(self.by[name].shape))

    def get(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        t = self.by[name]
        raw = np.array(t.data)
        if t.tensor_type == 0:  # F32
            arr = raw.astype(np.float32).reshape(self.logical_shape(name))
        else:
            rows = raw.shape[0]
            dq = dequantize(raw, t.tensor_type).astype(np.float32)
            arr = dq.reshape(rows, -1)
            assert tuple(arr.shape) == self.logical_shape(name), (name, arr.shape,
                                                                  self.logical_shape(name))
        self.cache[name] = arr
        return arr

    def get_t(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.get(name)).float()

    def _rows(self, name: str):
        t = self.by[name]
        raw = np.array(t.data)
        raw = raw.reshape(raw.shape[0], -1)   # first stored dim = rows (logical ne1)
        block_bytes = gguf.GGML_QUANT_SIZES[t.tensor_type][1]
        blocks_per_row = raw.shape[1] // block_bytes
        return t, raw, blocks_per_row

    def embed_rows(self, ids: list[int]) -> torch.Tensor:
        """Dequantise only the requested rows of token_embd."""
        t, raw, blocks_per_row = self._rows("token_embd.weight")
        rows = []
        for i in ids:
            blocks = raw[i].reshape(blocks_per_row, -1)
            rows.append(dequantize(blocks, t.tensor_type).astype(np.float32).reshape(-1))
        return torch.from_numpy(np.stack(rows)).float()

    def embed_logits(self, x: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        """Full-vocabulary logits against the (tied) embedding matrix, chunked."""
        t, raw, blocks_per_row = self._rows("token_embd.weight")
        vocab = raw.shape[0]
        out = []
        for r0 in range(0, vocab, chunk):
            r1 = min(vocab, r0 + chunk)
            blocks = raw[r0:r1].reshape(r1 - r0, blocks_per_row, -1)
            w = dequantize(blocks, t.tensor_type).astype(np.float32)
            w = torch.from_numpy(w).reshape(r1 - r0, -1)
            out.append(x @ w.T)
        return torch.cat(out, dim=1)

    def drop(self):
        self.cache.clear()


# ---------------------------------------------------------------------------
# Architecture (metadata driven; no model-name special cases)
# ---------------------------------------------------------------------------


class Config:
    def __init__(self, ts: GGUFTensors):
        kv = ts.fields
        arch = ts.arch
        get = lambda k, d=None: kv.get(f"{arch}.{k}", d)  # noqa: E731
        self.arch = arch
        self.n_layer = int(get("block_count"))
        self.hidden = int(get("embedding_length"))
        self.ffn = int(get("feed_forward_length"))
        self.n_head = int(get("attention.head_count"))
        self.eps = float(get("attention.layer_norm_rms_epsilon", 1e-6))
        self.ctx = int(get("context_length", 0) or 0)
        self.softcap = float(get("final_logit_softcapping", 0.0) or 0.0)
        self.attn_softcap = float(get("attention.logit_softcapping", 0.0) or 0.0)
        kv_arr = get("attention.head_count_kv")
        if isinstance(kv_arr, (list, tuple, np.ndarray)):
            self.n_kv = [int(x) for x in kv_arr]
        else:
            self.n_kv = [int(kv_arr or self.n_head)] * self.n_layer
        self.kl = int(get("attention.key_length", 0) or 0)
        self.vl = int(get("attention.value_length", 0) or 0)
        self.kl_swa = int(get("attention.key_length_swa", self.kl) or self.kl)
        self.vl_swa = int(get("attention.value_length_swa", self.vl) or self.vl)
        self.rope_base = float(get("rope.freq_base", 10000.0))
        self.rope_base_swa = float(get("rope.freq_base_swa", self.rope_base))
        self.rope_dim = int(get("rope.dimension_count", self.kl) or self.kl)
        self.rope_dim_swa = int(get("rope.dimension_count_swa", self.kl_swa) or self.kl_swa)
        self.window = int(get("attention.sliding_window", 0) or 0)
        pattern = get("attention.sliding_window_pattern")
        if isinstance(pattern, (list, tuple, np.ndarray)):
            self.is_swa = [bool(x) for x in pattern]
        else:
            self.is_swa = [False] * self.n_layer
        self.k_eq_v = bool(get("attention.k_eq_v", 0)) or ts.has("blk.0.attn_k_norm.weight")
        self.has_ple = int(get("embedding_length_per_layer_input", 0) or 0) > 0
        self.head_dim = [self.kl_swa if sw else self.kl for sw in self.is_swa]
        self.rope_theta = [self.rope_base_swa if sw else self.rope_base for sw in self.is_swa]
        self.rope_n = [self.rope_dim_swa if sw else self.rope_dim for sw in self.is_swa]

    def per_layer(self, i: int) -> dict:
        return {"swa": self.is_swa[i], "n_kv": self.n_kv[i], "d": self.head_dim[i],
                "theta": self.rope_theta[i], "n_rot": self.rope_n[i],
                "window": self.window if self.is_swa[i] else 0}


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float,
             plus1: bool = False) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True) + eps
    y = x.float() * torch.pow(var, -0.5)
    if w is not None:
        scale = (1.0 + w.float()) if plus1 else w.float()
        y = y * scale
    return y


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope(x: torch.Tensor, positions: torch.Tensor, n_rot: int, theta: float,
         partial: bool) -> torch.Tensor:
    """x: [T, heads, d]; rotary applied to the first n_rot dims (NEOX/rotate_half)."""
    d = x.shape[-1]
    n_rot = n_rot if partial else d
    n_rot = min(n_rot, d)
    inv = 1.0 / (theta ** (torch.arange(0, n_rot, 2, dtype=torch.float32) / n_rot))
    ang = positions.float()[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)[:, None, :]
    x_rot = x[..., :n_rot]
    x_pass = x[..., n_rot:]
    out = torch.cat([x_rot * cos + rotate_half(x_rot) * sin, x_pass], dim=-1)
    return out


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                       (x + 0.044715 * x.pow(3))))


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def forward(cfg: Config, ts: GGUFTensors, ids: list[int], opts) -> torch.Tensor:
    torch.set_num_threads(opts.threads)
    T = len(ids)
    pos = torch.arange(T, dtype=torch.long)
    x = ts.embed_rows(ids)
    if opts.embed_scale == "sqrt":
        x = x * math.sqrt(cfg.hidden)
    x = x.float()

    for i in range(cfg.n_layer):
        lp = cfg.per_layer(i)
        d, n_h, n_kv = lp["d"], cfg.n_head, lp["n_kv"]
        pre = f"blk.{i}."
        Ts = ts.get_t
        W = {
            "q": Ts(pre + "attn_q.weight"),
            "o": Ts(pre + "attn_output.weight"),
            "gate": Ts(pre + "ffn_gate.weight"),
            "up": Ts(pre + "ffn_up.weight"),
            "down": Ts(pre + "ffn_down.weight"),
            "attn_norm": Ts(pre + "attn_norm.weight"),
            "ffn_norm": Ts(pre + "ffn_norm.weight"),
            "q_norm": Ts(pre + "attn_q_norm.weight") if ts.has(pre + "attn_q_norm.weight") else None,
            "k_norm": Ts(pre + "attn_k_norm.weight") if ts.has(pre + "attn_k_norm.weight") else None,
            "post_attn": Ts(pre + "post_attention_norm.weight") if ts.has(pre + "post_attention_norm.weight") else None,
            "post_ffw": Ts(pre + "post_ffw_norm.weight") if ts.has(pre + "post_ffw_norm.weight") else None,
            "layer_scalar": (float(ts.get(pre + "layer_output_scale.weight")[0])
                             if ts.has(pre + "layer_output_scale.weight") else 1.0),
        }
        W["k"] = Ts(pre + "attn_k.weight")
        has_v = ts.has(pre + "attn_v.weight")
        if has_v:
            W["v"] = Ts(pre + "attn_v.weight")

        residual = x
        hn = rms_norm(x, W["attn_norm"], cfg.eps, opts.norm_mode == "plus1")
        q = (hn @ W["q"].T).view(T, n_h, d)
        if W["q_norm"] is not None:
            q = rms_norm(q, W["q_norm"], cfg.eps, opts.qk_norm_mode == "plus1")
        q = rope(q, pos, lp["n_rot"], lp["theta"], opts.rope_full)
        if has_v:
            k = (hn @ W["k"].T).view(T, n_kv, d)
            if W["k_norm"] is not None:
                k = rms_norm(k, W["k_norm"], cfg.eps, opts.qk_norm_mode == "plus1")
            k = rope(k, pos, lp["n_rot"], lp["theta"], opts.rope_full)
            v = (hn @ W["v"].T).view(T, n_kv, d)
            if opts.v_norm == "on":
                v = rms_norm(v, None, cfg.eps)
        else:
            k_raw = (hn @ W["k"].T).view(T, n_kv, d)
            k = rms_norm(k_raw, W["k_norm"], cfg.eps, opts.qk_norm_mode == "plus1") \
                if W["k_norm"] is not None else k_raw
            k = rope(k, pos, lp["n_rot"], lp["theta"], opts.rope_full)
            v = rms_norm(k_raw, None, cfg.eps) if opts.v_norm == "on" else k_raw

        rep = n_h // n_kv
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        scale = 1.0 if opts.attn_scale == "1.0" else 1.0 / math.sqrt(d)
        scores = torch.einsum("thd,shd->hts", q, kk) * scale
        idx = torch.arange(T)
        causal = idx[:, None] >= idx[None, :]
        if lp["window"]:
            causal &= (idx[:, None] - idx[None, :]) < lp["window"]
        scores = scores.masked_fill(~causal[None], float("-inf"))
        if opts.attn_softcap > 0:
            scores = torch.tanh(scores / opts.attn_softcap) * opts.attn_softcap
        attn = torch.softmax(scores.float(), dim=-1)
        ctx = torch.einsum("hts,shd->thd", attn, vv).reshape(T, n_h * d)
        o = ctx @ W["o"].T
        if opts.sandwich == "on" and W["post_attn"] is not None:
            o = rms_norm(o, W["post_attn"], cfg.eps, opts.norm_mode == "plus1")
        x = residual + o

        residual = x
        hn = rms_norm(x, W["ffn_norm"], cfg.eps, opts.norm_mode == "plus1")
        g = hn @ W["gate"].T
        u = hn @ W["up"].T
        act = gelu(g) if opts.act == "gelu" else F.silu(g)
        mlp = (act * u) @ W["down"].T
        if opts.sandwich == "on" and W["post_ffw"] is not None:
            mlp = rms_norm(mlp, W["post_ffw"], cfg.eps, opts.norm_mode == "plus1")
        x = residual + mlp
        if opts.layer_scalar == "on":
            x = x * W["layer_scalar"]
        ts.drop()
        if opts.verbose:
            print(f"  layer {i:2d} swa={int(lp['swa'])} d={d} n_kv={n_kv} "
                  f"|x|={float(x.abs().mean()):.4f}", flush=True)

    x = rms_norm(x, ts.get_t("output_norm.weight"), cfg.eps, opts.norm_mode == "plus1")
    logits = ts.embed_logits(x)
    if cfg.softcap and opts.softcap == "on":
        logits = torch.tanh(logits / cfg.softcap) * cfg.softcap
    return logits


# ---------------------------------------------------------------------------
# Compare with llama.cpp
# ---------------------------------------------------------------------------


def compare(logits: torch.Tensor, oracle_steps: list[dict]) -> None:
    lp = torch.log_softmax(logits[-1].float(), dim=-1)
    for step, o in enumerate(oracle_steps[:4]):
        top = o["top_logprobs"]
        print(f"step {step}: llama ids {[x['id'] for x in top[:5]]}")
        print(f"          llama logp {[round(x['logprob'], 4) for x in top[:5]]}")
        ref = [(int(t["id"]), round(float(lp[int(t["id"])]), 4)) for t in top[:5]]
        print(f"          ref   id/logp {ref}")
        diff = max(abs(float(lp[int(t["id"])]) - t["logprob"]) for t in top)
        print(f"          max |dlogp| (llama top-5) = {diff:.4f}")
        ref_top = torch.topk(lp, 5)
        match = int(ref_top.indices[0]) == top[0]["id"]
        print(f"          ref argmax {int(ref_top.indices[0])} "
              f"(llama {top[0]['id']})  {'MATCH' if match else 'MISMATCH'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--norm-mode", choices=["direct", "plus1"], default="direct")
    ap.add_argument("--qk-norm-mode", choices=["direct", "plus1"], default="direct")
    ap.add_argument("--attn-scale", choices=["1.0", "rsqrt"], default="1.0")
    ap.add_argument("--rope-full", type=int, default=1)
    ap.add_argument("--embed-scale", choices=["sqrt", "none"], default="sqrt")
    ap.add_argument("--v-norm", choices=["on", "off"], default="on")
    ap.add_argument("--sandwich", choices=["on", "off"], default="on")
    ap.add_argument("--layer-scalar", choices=["on", "off"], default="on")
    ap.add_argument("--softcap", choices=["on", "off"], default="on")
    ap.add_argument("--act", choices=["gelu", "silu"], default="gelu")
    ap.add_argument("--attn-softcap", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    opts = ap.parse_args()

    print(f"== llama.cpp oracle: prompt={opts.prompt!r} ==", flush=True)
    o = oracle(opts.prompt, opts.n)
    print("prompt ids:", o["ids"], "greedy:", o["gen"], flush=True)

    ts = GGUFTensors(Path(opts.model))
    cfg = Config(ts)
    print(f"arch={cfg.arch} layers={cfg.n_layer} hidden={cfg.hidden} heads={cfg.n_head} "
          f"kv={cfg.n_kv[:8]}... swa={cfg.is_swa[:8]}...", flush=True)
    print("opts:", {k: v for k, v in vars(opts).items() if k != "model"}, flush=True)
    t0 = time.time()
    logits = forward(cfg, ts, o["ids"], opts)
    print(f"ref forward done in {time.time() - t0:.1f}s, logits {tuple(logits.shape)}",
          flush=True)
    compare(logits, o["steps"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
