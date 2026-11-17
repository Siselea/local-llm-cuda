#!/usr/bin/env python
"""utilization_report.py -- turn measure_utilization.exe output + engine profiling logs into
utilisation/bubble statistics for the local inference engine.

Pipeline
--------
1. `tools/util_bin/measure_utilization.exe` spawns `build/test_gen.exe`, samples NVML
   (GPU util / VRAM / power / SM clock) plus the child's CPU time every ~10 ms, and
   timestamps every stderr line of the engine (`<t_ms>\t<line>`).
2. The engine is asked for its own instrumentation through existing env vars only:
     ENGINE_DBG_FWD=1        one "[fwd] pre pipe logits-dl total" line per forward
     ENGINE_DBG_LAYERMS=1    one "[layerms] layer i <wall> ms GPU|CPU sync/launch host" line
                             per layer for n<=2 tokens
     ENGINE_KPROF=1          per-kernel-kind GPU time (CUDA events) + GPU busy%
     ENGINE_DBG_CPUEXEC=1    host matvec ms/layer of the CPU expert pass
     ENGINE_PIPE_PROF=1      host-side launch/sync totals per forward
     ENGINE_PROFILE=1        MoE phase breakdown (only meaningful for n==2 forwards)
3. This script aligns the two timelines and reports, per run:
     * GPU utilisation time series (mean/median/max, share of samples below 30/10 %)
     * process CPU utilisation (% of wall * cores) and machine-wide CPU busy%
     * engine wall time per token from the [fwd] markers
     * per-layer table: wall, derived GPU time, host compute, idle
     * GPU attribution by kernel kind (calls classified against the layer/MoE/GPU-layer
       group sizes from the [placement] line), validated against the kprof total
     * prefix-layer differencing (`--layers N` runs) which measures per-class GPU time
       without any kernel-name heuristic
4. `--root DIR` summarises a whole measured suite (DIR/<model>/<mode>/...) into
   summary.json + summary.md.

No file outside tools/ is written; the engine is never modified.

Usage
-----
  # single run
  python tools/utilization_report.py --run tools/util_results/qwen3.5-9B/probe

  # whole suite (writes summary.json / summary.md next to the run dirs)
  python tools/utilization_report.py --root tools/util_results

Assumptions that are checked and printed, not silently trusted:
  * another engine process on the GPU (test_gen*/engine*/llama*) would corrupt the numbers;
    measures of `quiet_before_start`, `foreign_*` are printed as a validity block;
  * ENGINE_KPROF / ENGINE_DBG_LAYERMS need `--no-graph` for MoE models anyway and are
    *not* valid in CUDA-graph (dense) mode, so those runs are separate from the graph runs;
  * the derived per-layer GPU time is only claimed when the kernel-kind classification
    reproduces the measured kprof total within 2 %.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys
import re

# ---------------------------------------------------------------------------------
# model facts (from `engine info`, kept here so the report is reproducible)
# ---------------------------------------------------------------------------------
MODELS = {
    "qwen2.5-0.5b": dict(path="<MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                         ids="1000,1001", n_layers=24, n_moe=0, weights_gib=462.96 / 1024),
    "DeepSeek-Coder-V2-Lite": dict(
        path="<MODEL_DIR>/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf",
        ids="549,6077,280,7239", n_layers=27, n_moe=26, weights_gib=11.03),
    "qwen3.5-9B": dict(path="<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf",
                       ids="760", n_layers=32, n_moe=0, weights_gib=6.12),
    "qwen3.6-35B-A3B": dict(
        path="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf",
        ids="760", n_layers=40, n_moe=40, weights_gib=20.33),
}

GPU_NVLINK_FLOOR_GBPS = 330.0   # measured pure-read ceiling, RTX 3060 12GB (see Q6K-DP4A-RESULTS.md)
GPU_IDLE_W = 15.0
NUM_THREADS = 12                # 6 physical cores x 2 SMT


# ---------------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------------
def read_log(path):
    """-> list of (t_ms, text)"""
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            out.append((t, parts[2]))
    return out


def read_timeline(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split(",")
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split(",")
            if len(vals) != len(header):
                continue
            r = {}
            for k, v in zip(header, vals):
                try:
                    r[k] = float(v)
                except ValueError:
                    r[k] = float("nan")
            rows.append(r)
    return header, rows


def read_meta(run_dir):
    p = os.path.join(run_dir, "measurement.meta.json")
    if not os.path.exists(p):
        p = os.path.join(run_dir, "probe.meta.json")
    if not os.path.exists(p):
        p = os.path.join(run_dir, "graph.meta.json")
    if not os.path.exists(p):
        cands = glob.glob(os.path.join(run_dir, "*.meta.json"))
        p = cands[0] if cands else None
    if p and os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def find_files(run_dir):
    """Locate the (timeline, log, meta) triple inside a run dir regardless of prefix."""
    metas = sorted(glob.glob(os.path.join(run_dir, "*.meta.json")))
    if not metas:
        return None
    meta_path = metas[0]
    prefix = meta_path[: -len(".meta.json")]
    csv_path = prefix + ".timeline.csv"
    log_path = prefix + ".engine.log"
    if not (os.path.exists(csv_path) and os.path.exists(log_path)):
        return None
    return csv_path, log_path, meta_path, prefix


def in_window(rows, t0, t1):
    return [r for r in rows if t0 <= r["t_ms"] <= t1]


def col(rows, key):
    return [r[key] for r in rows]


def stats(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return dict(n=0)
    s = sorted(vals)

    def pct(p):
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * p
        lo = int(math.floor(k))
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return dict(n=len(vals), mean=sum(vals) / len(vals), median=statistics.median(vals),
                p10=pct(0.10), p90=pct(0.90), min=s[0], max=s[-1], stdev=statistics.pstdev(vals))


def share_below(vals, thr):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan")
    return 100.0 * sum(1 for v in vals if v < thr) / len(vals)


SPARK = " ▁▂▃▄▅▆▇█"


def sparkline(vals, lo=0.0, hi=100.0):
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(" ")
            continue
        f = (v - lo) / (hi - lo) if hi > lo else 0
        f = max(0.0, min(1.0, f))
        out.append(SPARK[1 + int(round(f * (len(SPARK) - 2)))])
    return "".join(out)


# ---------------------------------------------------------------------------------
# engine-log parsers
# ---------------------------------------------------------------------------------
RE_FWD = re.compile(r"\[fwd\] pre ([\d.]+)\s+pipe ([\d.]+)\s+logits-dl ([\d.]+)\s+total ([\d.]+)")
RE_PREFILL = re.compile(r"\[time\] prefill (\d+) tokens: ([\d.]+) ms")
RE_DECODE = re.compile(r"\[time\] decode ([\d.]+) ms/token")
RE_LAYERMS = re.compile(
    r"\[layerms\] layer\s+(\d+)\s+([\d.]+) ms\s+(GPU|CPU)\s+sync (\d+)/([\d.]+)\s+"
    r"launch (\d+)/([\d.]+)\s+host\s+([\d.]+) ms")
RE_KPROF_HEAD = re.compile(r"\[kprof\] total GPU time ([\d.]+) ms over (\d+) kernel kinds")
RE_KPROF_BUSY = re.compile(r"\[kprof\] wall \(decoded tokens\) ([\d.]+) ms -> GPU busy ([\d.]+)%")
RE_KPROF_KERNEL = re.compile(
    r"\[kprof\]\s+([\d.]+) ms\s+(\d+) calls\s+(\S+)\s+([\d.]+)%")
RE_PLACEMENT = re.compile(r"\[placement\] gpu_moe_layers=(-?\d+)/(-?\d+)")
RE_VRAM = re.compile(r"\[vram\] post-load used=(\d+) MiB free=(\d+) MiB")
RE_CPUEXEC = re.compile(r"\[cpuexec\] batched host matvec avg ([\d.]+) ms over (\d+) layers")
RE_CPULAYER = re.compile(r"\[cpulayer\] sync ([\d.]+) ms, D2H ([\d.]+) ms")
RE_PIPE_1 = re.compile(r"\[pipe\] n=(\d+) layers=(\d+) wall ([\d.]+) ms, cpu-launch ([\d.]+) ms, "
                       r"syncs (\d+) = ([\d.]+) ms")
RE_PIPE_2 = re.compile(r"\[pipe\] launches (\d+) = ([\d.]+) ms, mla ([\d.]+) ms, moe ([\d.]+) ms")
RE_MOEPROF = re.compile(r"\[moe-prof\] layer (\d+): router ([\d.]+), experts ([\d.]+), "
                        r"gap ([\d.]+), total ([\d.]+)")
RE_MOEPROF2 = re.compile(r"\[moe-prof\]\s+layer (\d+) \(n=(\d+)\): experts ([\d.]+) ms, "
                         r"sh gate\+up ([\d.]+) ms")
RE_ATTNPROF = re.compile(r"\[attn-prof\] layer (\d+) mla ([\d.]+) ms")
RE_GPUEND = re.compile(r"\[warn\] parsed in ([\d.]+) ms")


def parse_engine_log(lines):
    """Extract every instrumentation record from the timestamped engine log."""
    d = dict(fwd=[], prefill=None, decode=None, layerms_blocks=[], kprof=None,
             placement=None, vram=None, cpuexec=[], cpulayer=[], pipe=[], moeprof=[],
             attnprof=[], load_ms=None)
    cur_block = []
    for t, text in lines:
        m = RE_FWD.search(text)
        if m:
            d["fwd"].append(dict(t=t, pre=float(m.group(1)), pipe=float(m.group(2)),
                                 logits=float(m.group(3)), total=float(m.group(4))))
            continue
        m = RE_PREFILL.search(text)
        if m:
            d["prefill"] = dict(t=t, tokens=int(m.group(1)), ms=float(m.group(2)))
            continue
        m = RE_DECODE.search(text)
        if m:
            d["decode"] = dict(t=t, ms_per_token=float(m.group(1)))
            continue
        m = RE_LAYERMS.search(text)
        if m:
            layer = int(m.group(1))
            if layer == 0 and cur_block:
                d["layerms_blocks"].append(cur_block)
                cur_block = []
            cur_block.append(dict(layer=layer, wall=float(m.group(2)), cls=m.group(3),
                                  sync=int(m.group(4)), sync_ms=float(m.group(5)),
                                  launch=int(m.group(6)), launch_ms=float(m.group(7)),
                                  host=float(m.group(8))))
            continue
        if "RE_LAYERMS_END" in text:
            if cur_block:
                d["layerms_blocks"].append(cur_block)
                cur_block = []
            continue
        m = RE_KPROF_HEAD.search(text)
        if m:
            d["kprof"] = dict(total_gpu=float(m.group(1)), n_kinds=int(m.group(2)),
                              kernels=[], wall=None, busy=None)
            continue
        m = RE_KPROF_BUSY.search(text)
        if m and d["kprof"]:
            d["kprof"]["wall"] = float(m.group(1))
            d["kprof"]["busy"] = float(m.group(2))
            continue
        m = RE_KPROF_KERNEL.search(text)
        if m and d["kprof"] is not None:
            d["kprof"]["kernels"].append(dict(ms=float(m.group(1)), calls=int(m.group(2)),
                                              name=m.group(3), pct=float(m.group(4))))
            continue
        m = RE_PLACEMENT.search(text)
        if m:
            d["placement"] = dict(gpu_moe=int(m.group(1)), total_moe=int(m.group(2)))
            continue
        m = RE_VRAM.search(text)
        if m:
            d["vram"] = dict(used_mib=int(m.group(1)), free_mib=int(m.group(2)))
            continue
        m = RE_CPUEXEC.search(text)
        if m:
            d["cpuexec"].append(dict(t=t, ms=float(m.group(1)), layers=int(m.group(2))))
            continue
        m = RE_CPULAYER.search(text)
        if m:
            d["cpulayer"].append(dict(t=t, sync=float(m.group(1)), d2h=float(m.group(2))))
            continue
        m = RE_PIPE_1.search(text)
        if m:
            d["pipe"].append(dict(t=t, n=int(m.group(1)), layers=int(m.group(2)),
                                  wall=float(m.group(3)), cpu_launch=float(m.group(4)),
                                  syncs=int(m.group(5)), sync_ms=float(m.group(6)),
                                  launches=None, launch_ms=None, mla=None, moe=None))
            continue
        m = RE_PIPE_2.search(text)
        if m and d["pipe"]:
            p = d["pipe"][-1]
            p["launches"] = int(m.group(1))
            p["launch_ms"] = float(m.group(2))
            p["mla"] = float(m.group(3))
            p["moe"] = float(m.group(4))
            continue
        m = RE_MOEPROF.search(text)
        if m:
            d["moeprof"].append(dict(t=t, layer=int(m.group(1)), router=float(m.group(2)),
                                     experts=float(m.group(3)), gap=float(m.group(4)),
                                     total=float(m.group(5))))
            continue
        m = RE_MOEPROF2.search(text)
        if m:
            d["moeprof"].append(dict(t=t, layer=int(m.group(1)), n=int(m.group(2)),
                                     experts=float(m.group(3)), sh_gu=float(m.group(4))))
            continue
        m = RE_ATTNPROF.search(text)
        if m:
            d["attnprof"].append(dict(t=t, layer=int(m.group(1)), mla=float(m.group(2))))
            continue
    if cur_block:
        d["layerms_blocks"].append(cur_block)
    return d


# ---------------------------------------------------------------------------------
# kernel-kind -> layer-class attribution
# ---------------------------------------------------------------------------------
def attribute_kernels(kprof, n_layers, n_moe, n_gpu_moe):
    """Assign each kprof kernel kind to a launch group using its call count.

    The engine launches the same decode kernels once (or a fixed number of times) per
    layer, so the call count is a multiple of the group size:
        n_layers   -- kernels launched in every layer (norm, rope, attention, ...)
        n_moe      -- kernels launched in every MoE layer (router, shared expert, ...)
        n_gpu_moe  -- kernels launched only in GPU-resident MoE layers (expert GEMV)
        1          -- run-level kernels (final norm, lm_head)
    Groups whose size does not divide the call count are skipped.  When several divide,
    the *largest* wins; the two ambiguous cases that this could hide (a kernel launched
    n_layers/G times per layer in G groups) are reported through `ambiguous`.
    """
    groups = []
    if n_layers > 0:
        groups.append(("all_layers", n_layers))
    if n_moe > 0 and n_moe != n_layers:
        groups.append(("moe_layers", n_moe))
    if n_gpu_moe > 0 and n_gpu_moe not in (n_layers, n_moe):
        groups.append(("gpu_layers", n_gpu_moe))
    groups.append(("run_level", 1))
    out = []
    ambiguous = []
    for k in kprof.get("kernels", []):
        calls = k["calls"]
        cand = [g for g in groups if g[1] > 0 and calls % g[1] == 0]
        if not cand:
            cand = [("run_level", 1)]
        cand.sort(key=lambda g: g[1], reverse=True)
        group, size = cand[0]
        if len(cand) > 1 and cand[1][1] > 1 and cand[1][1] != size:
            # e.g. 120 calls = 3/layer over 40 layers  OR  8/layer over 15 GPU layers
            if size % cand[1][1] == 0 or cand[1][1] % size == 0:
                ambiguous.append(dict(name=k["name"], calls=calls, chosen=group,
                                      alternative=cand[1][0]))
        out.append(dict(k, group=group, per_layer=calls // size, gpu_per_layer_ms=k["ms"] / size,
                        gpu_per_call_ms=k["ms"] / calls))
    assigned = sum(k["ms"] for k in out)
    return out, ambiguous, assigned


class Run:
    """One measured run: timeline + engine log + derived statistics."""

    def __init__(self, run_dir, model=None):
        self.dir = run_dir
        files = find_files(run_dir)
        if not files:
            raise SystemExit("no measurement triple in %s" % run_dir)
        self.csv_path, self.log_path, self.meta_path, self.prefix = files
        self.meta = read_meta(run_dir)
        self.header, self.rows = read_timeline(self.csv_path)
        self.lines = read_log(self.log_path)
        self.engine = parse_engine_log(self.lines)
        self.name = os.path.basename(run_dir.rstrip("/\\"))
        self.model_dir = os.path.basename(os.path.dirname(run_dir.rstrip("/\\")))
        self.model = model or MODELS.get(self.model_dir)
        self.cmd = self.meta.get("command", "")
        self._decode_window()
        self._token_windows()
        self._class_timeline()

    # ---- windows ---------------------------------------------------------------
    def _decode_window(self):
        e = self.engine
        t_end = self.rows[-1]["t_ms"] if self.rows else 0.0
        self.t_dec_start = None
        self.t_dec_end = None
        if e["prefill"]:
            self.t_dec_start = e["prefill"]["t"]
        if e["decode"]:
            # the decode summary is printed after the last token -> window ends at the
            # last [fwd] line before it (fall back to the summary timestamp)
            fwds = [f["t"] for f in e["fwd"] if f["t"] < e["decode"]["t"]]
            self.t_dec_end = fwds[-1] if fwds else e["decode"]["t"]
        if self.t_dec_start is None:  # fall back: from the first fwd line to the last
            if e["fwd"]:
                self.t_dec_start = e["fwd"][0]["t"]
                self.t_dec_end = e["fwd"][-1]["t"]
            else:
                self.t_dec_start, self.t_dec_end = 0.0, t_end
        if self.t_dec_end is None:
            # no [time] decode line: the run was killed (engine hang / timeout). The last
            # completed forward is the end of the usable data.
            fwds = [f["t"] for f in e["fwd"] if f["t"] > self.t_dec_start]
            if fwds:
                self.t_dec_end = fwds[-1]
        if self.t_dec_end is None or self.t_dec_end <= self.t_dec_start:
            self.t_dec_end = max(t_end, self.t_dec_start)

    def _token_windows(self):
        """Split the decode window at [fwd] markers; drop the CUDA-graph capture token."""
        e = self.engine
        times = [f["t"] for f in e["fwd"] if self.t_dec_start <= f["t"] <= self.t_dec_end + 1e-6]
        starts = [self.t_dec_start] + times[:-1]
        ends = times
        if len(starts) != len(ends):
            starts = starts[: len(ends)]
        toks = []
        for i, (a, b) in enumerate(zip(starts, ends)):
            toks.append(dict(i=i, t0=a, t1=b, wall=b - a))
        walls = [t["wall"] for t in toks]
        self.capture_tokens = []
        self.outlier_tokens = []
        if walls:
            med = statistics.median(walls)
            # CUDA-graph capture can only happen on the first n==1 forward (token 0)
            if med > 0 and toks and toks[0]["wall"] > 1.5 * med:
                self.capture_tokens.append(toks[0]["i"])
            for t in toks:
                if med > 0 and t["wall"] > 2.5 * med and t["i"] not in self.capture_tokens:
                    self.outlier_tokens.append(t["i"])
        self.tokens = toks

    # ---- statistics ------------------------------------------------------------
    def decode_rows(self, drop_capture=True):
        lo, hi = self.t_dec_start, self.t_dec_end
        rows = in_window(self.rows, lo, hi)
        if drop_capture and self.capture_tokens:
            drop = set(self.capture_tokens)
            kept = []
            for r in rows:
                bad = False
                for i in drop:
                    t = self.tokens[i]
                    if t["t0"] <= r["t_ms"] <= t["t1"]:
                        bad = True
                        break
                if not bad:
                    kept.append(r)
            rows = kept
        return rows

    def util_stats(self, drop_capture=True):
        rows = self.decode_rows(drop_capture)
        util = col(rows, "gpu_util")
        power = col(rows, "gpu_power_w")
        sm = col(rows, "sm_mhz")
        proc = col(rows, "proc_cpu_pct")
        sys = col(rows, "sys_cpu_busy_pct")
        st = dict(
            gpu_util=stats(util), gpu_power=stats(power), sm_mhz=stats(sm),
            proc_cpu=stats(proc), sys_cpu=stats(sys),
            frac_util_lt30=share_below(util, 30.0), frac_util_lt10=share_below(util, 10.0),
            frac_util_lt5=share_below(util, 5.0),
            frac_util_ge80=100.0 - share_below(util, 80.0),
        )
        return st

    def _class_timeline(self):
        """Reconstruct absolute layer windows from [layerms] blocks in the decode phase.

        Line timestamps alone are too coarse (the log reader gets several lines per
        read), so the engine's own per-layer wall times are summed from the token
        start given by the [fwd] marker.  Each layer window is labelled GPU/CPU; the
        NVML samples inside it are averaged per class.
        """
        e = self.engine
        self.layer_windows = []
        self.class_stats = {}
        self.layer_util = {}
        if not e["layerms_blocks"] or not self.tokens:
            return
        # decode blocks: those whose first line was printed inside the decode window
        blocks = [b for b in e["layerms_blocks"]
                  if b and b[0]["layer"] == 0]
        # keep the last len(tokens) blocks (warm-up/prefill blocks are older)
        if len(blocks) > len(self.tokens):
            blocks = blocks[-len(self.tokens):]
        if not blocks:
            return
        for k, b in enumerate(blocks):
            if k >= len(self.tokens):
                break
            tok = self.tokens[k]
            total_wall = sum(x["wall"] for x in b) or 1.0
            scale = (tok["t1"] - tok["t0"]) / total_wall
            cur = tok["t0"]
            for x in b:
                dur = x["wall"] * scale
                self.layer_windows.append(dict(t0=cur, t1=cur + dur, layer=x["layer"],
                                               cls=x["cls"], wall=x["wall"] * scale))
                cur += dur
        rows = self.decode_rows()
        per_layer = {}
        for w in self.layer_windows:
            inside = in_window(rows, w["t0"], w["t1"])
            if inside:
                per_layer.setdefault((w["cls"], w["layer"]), []).extend(col(inside, "gpu_util"))
        self.layer_util = {k: statistics.mean(v) for k, v in per_layer.items()}
        # aggregate NVML util per class, weighting each layer window equally
        for cls in ("GPU", "CPU"):
            per_class = []
            for w in self.layer_windows:
                if w["cls"] != cls:
                    continue
                inside = in_window(rows, w["t0"], w["t1"])
                util = col(inside, "gpu_util")
                if util:
                    per_class.append(sum(util) / len(util))
            if per_class:
                self.class_stats[cls] = dict(
                    n=len(per_class),
                    util_mean=statistics.mean(per_class),
                    util_median=statistics.median(per_class),
                )

    # ---- derived per-layer decomposition ---------------------------------------
    def layer_table(self):
        e = self.engine
        if not self.engine["layerms_blocks"]:
            return None
        # average each layer index over the decode-token blocks (skip the first block,
        # which is the 2-token warm-up, and the prefill block when it exists)
        blocks = [b for b in e["layerms_blocks"] if b and b[0]["layer"] == 0]
        if not blocks:
            return None
        tok_blocks = blocks
        if e["prefill"]:
            # decode blocks start at the first block whose first line timestamp > prefill
            tok_blocks = [b for b in blocks if self.t_dec_start is not None]
        # choose the blocks that belong to token windows
        if self.tokens:
            tok_blocks = blocks[-len(self.tokens):] if len(blocks) >= len(self.tokens) else blocks
        if not tok_blocks:
            tok_blocks = blocks
        n = min(len(b) for b in tok_blocks)
        layers = []
        for i in range(n):
            xs = [b[i] for b in tok_blocks]
            cls = xs[0]["cls"]
            layers.append(dict(layer=i, cls=cls,
                               wall=statistics.mean(x["wall"] for x in xs),
                               sync=statistics.mean(x["sync"] for x in xs),
                               sync_ms=statistics.mean(x["sync_ms"] for x in xs),
                               launch=statistics.mean(x["launch"] for x in xs),
                               launch_ms=statistics.mean(x["launch_ms"] for x in xs),
                               host=statistics.mean(x["host"] for x in xs)))
        return dict(layers=layers, n_blocks=len(tok_blocks))

    def host_matvec_ms(self):
        """CPU expert matvec ms/layer from [cpuexec] prints (median of the prints)."""
        v = [c["ms"] for c in self.engine["cpuexec"]]
        if not v:
            return None
        # the first prints cover the warm-up 2-token batch; the decode prints are the last
        tail = v[2:] if len(v) > 3 else v
        return dict(median=statistics.median(tail), n=len(v), all=v)


# ---------------------------------------------------------------------------------
# single-run report
# ---------------------------------------------------------------------------------
def fmt(v, nd=2, dash="-"):
    if v is None:
        return dash
    if isinstance(v, float) and math.isnan(v):
        return dash
    return ("%." + str(nd) + "f") % v


def run_report(run):
    e = run.engine
    st = run.util_stats()
    ut = st["gpu_util"]
    out = {}
    out["dir"] = run.dir
    out["model"] = run.model_dir
    out["mode"] = run.name
    out["command"] = run.cmd
    out["meta"] = run.meta
    out["decode_window_ms"] = run.t_dec_end - run.t_dec_start
    out["engine"] = dict(
        prefill_ms=e["prefill"]["ms"] if e["prefill"] else None,
        prefill_tokens=e["prefill"]["tokens"] if e["prefill"] else None,
        decode_ms_per_token=e["decode"]["ms_per_token"] if e["decode"] else None,
        placement=e["placement"],
        vram=e["vram"],
        n_fwd_lines=len(e["fwd"]),
        token_walls=[round(t["wall"], 3) for t in run.tokens],
        capture_tokens=run.capture_tokens,
        host_matvec=run.host_matvec_ms(),
    )
    # steady-state per-token wall: median over tokens with the CUDA-graph capture token
    # dropped.  The engine's own [time] decode average includes that capture token, which
    # inflates it by (capture cost)/(n_tokens); the median is the fair steady-state value.
    steady = [t["wall"] for t in run.tokens if t["i"] not in run.capture_tokens]
    if steady:
        out["engine"]["token_wall_median"] = statistics.median(steady)
        out["engine"]["token_wall_min"] = min(steady)
        out["engine"]["token_wall_p90"] = sorted(steady)[max(0, int(round(0.9 * len(steady))) - 1)]
        out["engine"]["token_wall_n"] = len(steady)
    out["gpu"] = dict(
        util_mean=ut.get("mean"), util_median=ut.get("median"), util_p10=ut.get("p10"),
        util_p90=ut.get("p90"), util_max=ut.get("max"), samples=ut.get("n"),
        frac_lt30=st["frac_util_lt30"], frac_lt10=st["frac_util_lt10"],
        frac_lt5=st["frac_util_lt5"], frac_ge80=st["frac_util_ge80"],
        power_mean=st["gpu_power"].get("mean"), power_max=st["gpu_power"].get("max"),
        power_p10=st["gpu_power"].get("p10"), sm_mhz_mean=st["sm_mhz"].get("mean"),
        sm_mhz_median=st["sm_mhz"].get("median"), sm_mhz_max=st["sm_mhz"].get("max"),
        sm_mhz_min=st["sm_mhz"].get("min"),
        ambient_util=run.meta.get("ambient_gpu_util"),
        ambient_power_w=run.meta.get("ambient_gpu_power_w"),
    )
    out["cpu"] = dict(
        proc_pct_mean=st["proc_cpu"].get("mean"),
        proc_pct_median=st["proc_cpu"].get("median"),
        proc_cores_mean=(st["proc_cpu"].get("mean") or 0) / 100.0,
        proc_frac_of_machine=(st["proc_cpu"].get("mean") or 0) / 100.0 / NUM_THREADS,
        sys_busy_mean=st["sys_cpu"].get("mean"),
    )
    out["class_stats"] = run.class_stats
    # validity / contamination block (a concurrent engine on the same GPU invalidates
    # everything below, so it is surfaced as a first-class field)
    mt = run.meta
    rows = run.decode_rows()
    util = col(rows, "gpu_util")
    transitions = 0
    for i in range(1, len(util)):
        if not (isinstance(util[i], float) and math.isnan(util[i])) and \
           not (isinstance(util[i - 1], float) and math.isnan(util[i - 1])) and \
           util[i] != util[i - 1]:
            transitions += 1
    out["validity"] = dict(
        quiet_before_start=mt.get("quiet_before_start"),
        foreign_before_start=mt.get("foreign_before_start"),
        foreign_during_run_hits=mt.get("foreign_during_run_hits"),
        foreign_during_run_pids=mt.get("foreign_during_run_pids"),
        foreign_after_run=mt.get("foreign_after_run"),
        child_exit_code=mt.get("child_exit_code"),
        timed_out=mt.get("timed_out"),
        interval_mean_ms=mt.get("interval_mean_ms"),
        interval_max_ms=mt.get("interval_max_ms"),
        nvml_distinct_transitions=transitions,
        decode_samples=len(rows),
        contaminated=bool((mt.get("foreign_during_run_hits") or 0) > 0
                          or (mt.get("foreign_before_start") or 0) > 0),
    )
    if e["kprof"]:
        kp = e["kprof"]
        out["kprof"] = dict(total_gpu_ms=kp["total_gpu"], wall_ms=kp["wall"],
                            busy_pct=kp["busy"], kernels=kp["kernels"])
    # per-layer table + attribution
    lt = run.layer_table()
    if lt:
        n_layers = len(lt["layers"])
        out["n_layers"] = n_layers
        class_wall = {}
        for cls in ("GPU", "CPU"):
            xs = [x for x in lt["layers"] if x["cls"] == cls]
            if xs:
                class_wall[cls] = dict(
                    n=len(xs), wall=statistics.mean(x["wall"] for x in xs),
                    sync_ms=statistics.mean(x["sync_ms"] for x in xs),
                    launch_ms=statistics.mean(x["launch_ms"] for x in xs),
                    host=statistics.mean(x["host"] for x in xs))
        out["class_wall"] = class_wall
        hm = run.host_matvec_ms()
        host_cpu_layer = hm["median"] if hm else 0.0
        placement = e["placement"] or {}
        n_moe = placement.get("total_moe", run.model["n_moe"] if run.model else 0)
        n_gpu_moe = placement.get("gpu_moe", n_layers)
        if n_moe == 0:
            n_moe = n_layers
            n_gpu_moe = n_layers
        if e["kprof"]:
            attributed, ambiguous, assigned = attribute_kernels(
                e["kprof"], n_layers, n_moe, n_gpu_moe)
            for k in attributed:
                if k["group"] in ("gpu_layers",):
                    k["gpu_per_layer_ms"] = k["ms"] / max(1, n_gpu_moe)
            uni = sum(k["gpu_per_layer_ms"] for k in attributed if k["group"] == "all_layers")
            gpu_only = sum(k["gpu_per_layer_ms"] for k in attributed if k["group"] == "gpu_layers")
            runlevel = sum(k["ms"] for k in attributed if k["group"] == "run_level")
            out["attribution"] = dict(
                n_layers=n_layers, n_moe=n_moe, n_gpu_moe=n_gpu_moe,
                uniform_per_layer_ms=uni, gpu_layer_extra_ms=gpu_only,
                run_level_ms=runlevel, assigned_total_ms=assigned,
                kprof_total_ms=e["kprof"]["total_gpu"],
                ambiguous=ambiguous,
                gpu_layer_total_gpu_ms=uni + gpu_only,
                cpu_layer_total_gpu_ms=uni,
                kernels=[dict(name=k["name"], ms=k["ms"], calls=k["calls"], group=k["group"],
                              per_layer=k["per_layer"],
                              gpu_per_layer_ms=k["gpu_per_layer_ms"]) for k in attributed],
            )
            # per-layer idle = wall - GPU - host
            rows = []
            for x in lt["layers"]:
                cls = x["cls"]
                is_gpu = cls == "GPU"
                gpu = uni + (gpu_only if is_gpu else 0.0)
                host = 0.0 if is_gpu else host_cpu_layer
                idle = x["wall"] - gpu - host
                rows.append(dict(layer=x["layer"], cls=cls, wall=x["wall"], gpu=gpu, host=host,
                                 idle=idle, sync_ms=x["sync_ms"], launch_ms=x["launch_ms"],
                                 util=run.layer_util.get((cls, x["layer"]))))
            out["layer_rows"] = rows
            out["layer_util_map"] = {"%s:%d" % k: v for k, v in run.layer_util.items()}
            out["host_matvec_cpu_layer"] = host_cpu_layer
        else:
            # layerms-only run (no ENGINE_KPROF): wall + host matvec are still measured, the
            # GPU part has to come from a prefix-differenced run; keep the table with gpu=-1
            rows = []
            for x in lt["layers"]:
                is_gpu = x["cls"] == "GPU"
                host = 0.0 if is_gpu else host_cpu_layer
                rows.append(dict(layer=x["layer"], cls=x["cls"], wall=x["wall"], gpu=-1.0,
                                 host=host, idle=float("nan"), sync_ms=x["sync_ms"],
                                 launch_ms=x["launch_ms"],
                                 util=run.layer_util.get((x["cls"], x["layer"]))))
            out["layer_rows"] = rows
            out["layer_util_map"] = {"%s:%d" % k: v for k, v in run.layer_util.items()}
            out["host_matvec_cpu_layer"] = host_cpu_layer
            out["idle_totals"] = dict(
                wall=sum(r["wall"] for r in rows),
                gpu=sum(r["gpu"] for r in rows),
                host=sum(r["host"] for r in rows),
                idle=sum(r["idle"] for r in rows),
                gpu_classes={c: sum(r["gpu"] for r in rows if r["cls"] == c) for c in ("GPU", "CPU")},
                wall_classes={c: sum(r["wall"] for r in rows if r["cls"] == c)
                              for c in ("GPU", "CPU")},
                n_classes={c: sum(1 for r in rows if r["cls"] == c) for c in ("GPU", "CPU")},
            )
    # time series (per-token)
    toks = []
    for t in run.tokens:
        rows = in_window(run.rows, t["t0"], t["t1"])
        util = col(rows, "gpu_util")
        proc = col(rows, "proc_cpu_pct")
        toks.append(dict(i=t["i"], wall=t["wall"], util_mean=statistics.mean(util) if util else None,
                         util_spark=sparkline(util), power=statistics.mean(col(rows, "gpu_power_w"))
                         if rows else None,
                         proc=statistics.mean(proc) if proc else None))
    out["tokens"] = toks
    # time series over the decode window for the text timeline in the report.
    # NVML's utilisation counter is updated by the driver at a coarse cadence (~100 ms),
    # so bins smaller than that would only show the counter's quantisation; 50 ms bins
    # are used as a compromise (and the bin count is capped at 80).
    lo, hi = run.t_dec_start, run.t_dec_end
    if hi > lo:
        bin_ms = max(50.0, (hi - lo) / 80.0)
        nb = max(1, int(round((hi - lo) / bin_ms)))
        util_bins, power_bins, proc_bins = [], [], []
        last_util, last_power = float("nan"), float("nan")
        for i in range(nb):
            a, b = lo + i * bin_ms, lo + (i + 1) * bin_ms
            inside = in_window(run.rows, a, b)
            u = col(inside, "gpu_util")
            p = col(inside, "gpu_power_w")
            c = col(inside, "proc_cpu_pct")
            uu = sum(u) / len(u) if u else float("nan")
            pp = sum(p) / len(p) if p else float("nan")
            if math.isnan(uu):
                uu = last_util
            if math.isnan(pp):
                pp = last_power
            last_util, last_power = uu, pp
            util_bins.append(uu)
            power_bins.append(pp)
            proc_bins.append(sum(c) / len(c) if c else float("nan"))
        out["series"] = dict(bin_ms=bin_ms, util=util_bins, power=power_bins, proc=proc_bins)
    return out


# ---------------------------------------------------------------------------------
# suite summary
# ---------------------------------------------------------------------------------
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def analyse_suite(root):
    runs = {}
    for meta in glob.glob(os.path.join(root, "*", "*", "*.meta.json")):
        run_dir = os.path.dirname(meta)
        files = find_files(run_dir)
        if not files:
            continue
        model_dir = os.path.basename(os.path.dirname(run_dir))
        mode = os.path.basename(run_dir.rstrip("/\\"))
        try:
            r = Run(run_dir)
        except SystemExit as ex:
            print("skip", run_dir, ex)
            continue
        rep = run_report(r)
        with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        runs.setdefault(model_dir, {})[mode] = rep
    return runs


def suite_markdown(runs):
    """Render the whole measured suite as markdown (tables only, no plots)."""
    L = []
    L.append("# utilisation / bubble measurements (auto-generated)\n")
    L.append("Generated by `python tools/utilization_report.py --root tools/util_results`.\n")
    L.append("GPU utilisation comes from NVML at 10 ms; the engine's own GPU busy share "
             "comes from CUDA events (`ENGINE_KPROF`).  The Windows desktop keeps an "
             "ambient ~25 %% NVML 'utilisation' (compositor/CEF apps) even when idle, so "
             "the ambient column matters when reading the raw util numbers.\n")
    ordered = ["qwen2.5-0.5b", "qwen3.5-9B", "DeepSeek-Coder-V2-Lite",
               "qwen3.6-35B-A3B"]
    model_order = [m for m in ordered if m in runs] + [m for m in sorted(runs) if m not in ordered]

    # ---------------- validity ----------------
    L.append("\n## validity / contamination\n")
    rows = []
    for model in model_order:
        for mode in sorted(runs[model]):
            r = runs[model][mode]
            v = r.get("validity", {})
            mt = r.get("meta", {})
            rows.append([model, mode,
                         "y" if v.get("quiet_before_start") else "N",
                         "KILLED" if v.get("timed_out") else "ok",
                         v.get("foreign_before_start", "?"),
                         v.get("foreign_during_run_hits", "?"),
                         v.get("foreign_after_run", "?"),
                         fmt(mt.get("ambient_gpu_util"), 1),
                         v.get("decode_samples", "?"),
                         fmt(v.get("interval_mean_ms"), 1),
                         fmt(v.get("interval_max_ms"), 1),
                         "YES" if v.get("contaminated") else "no"])
    L.append(md_table(["model", "run", "quiet@start", "child", "foreign@start",
                       "foreign@during", "foreign@end", "ambient util%", "decode samples",
                       "dt mean ms", "dt max ms", "contaminated"], rows))

    # ---------------- global ----------------
    L.append("\n## global: wall, NVML GPU utilisation, process CPU\n")
    rows = []
    for model in model_order:
        for mode in sorted(runs[model]):
            r = runs[model][mode]
            g = r.get("gpu", {})
            c = r.get("cpu", {})
            kp = r.get("kprof")
            e = r.get("engine", {})
            rows.append([model, mode,
                         fmt(e.get("token_wall_median")),
                         fmt(e.get("decode_ms_per_token")),
                         fmt(g.get("util_mean"), 1),
                         fmt(g.get("util_median"), 1),
                         fmt(g.get("frac_lt30"), 1),
                         fmt(g.get("frac_lt10"), 1),
                         fmt(g.get("power_mean"), 1),
                         fmt(g.get("sm_mhz_mean"), 0),
                         fmt(c.get("proc_pct_mean"), 1),
                         fmt(c.get("proc_cores_mean"), 2),
                         fmt(c.get("sys_busy_mean"), 1),
                         fmt(kp["busy_pct"], 1) if kp else "-"])
    L.append(md_table(["model", "run", "steady decode ms/tok (median, capture token excluded)",
                       "engine avg decode ms/tok", "gpu util mean%", "median%",
                       "util<30% (of time)", "util<10%", "power W", "SM MHz", "proc CPU%",
                       "proc cores", "machine CPU%", "kprof GPU busy%"], rows))

    # ---------------- per-class NVML ----------------
    L.append("\n## NVML util binned into layer windows (probe run)\n")
    rows = []
    for model in model_order:
        r = runs[model].get("probe")
        if not r:
            continue
        for cls, v in sorted((r.get("class_stats") or {}).items()):
            rows.append([model, cls, fmt(v["util_mean"], 1), fmt(v["util_median"], 1), v["n"]])
    if rows:
        L.append(md_table(["model", "layer class", "util mean%", "util median%",
                           "layer windows"], rows))

    # ---------------- per-class GPU time by prefix differencing ----------------
    L.append("\n## measured GPU time per layer class (prefix-layer differencing)\n")
    L.append("Runs with `--layers N` execute only the first N layers, so "
             "`T(N) = N*p + H` for a uniform prefix; the differences below are direct "
             "measurements of the per-class GPU time without any kernel-name heuristic "
             "(T = `ENGINE_KPROF` total GPU ms of the last decode token, H = final "
             "norm+lm_head).  Wall numbers are the engine's decode ms/token.\n")
    rows = []
    for model in model_order:
        modes = runs[model]
        probe = modes.get("probe")
        p1 = modes.get("prefix1")
        if not (probe and p1 and probe.get("kprof") and p1.get("kprof")):
            continue
        md = MODELS.get(model, {})
        n = md.get("n_layers") or probe.get("n_layers")
        T1 = p1["kprof"]["total_gpu_ms"]
        Tn = probe["kprof"]["total_gpu_ms"]
        W1 = p1["engine"].get("decode_ms_per_token")
        Wn = probe["engine"].get("decode_ms_per_token")
        n_gpu = (probe.get("engine", {}).get("placement") or {}).get("gpu_moe", 0)
        pn = modes.get("prefixN")
        if model == "qwen3.6-35B-A3B" and pn and pn.get("kprof"):
            T15 = pn["kprof"]["total_gpu_ms"]
            W15 = pn["engine"].get("decode_ms_per_token")
            p_gpu = (T15 - T1) / 14.0
            c_cpu = (Tn - T15) / 25.0
            H = T1 - p_gpu
            w_gpu = (W15 - W1) / 14.0
            w_cpu = (Wn - W15) / 25.0
            rows.append([model, fmt(p_gpu, 3), fmt(c_cpu, 3), fmt(H, 3),
                         fmt(w_gpu, 3), fmt(w_cpu, 3),
                         "n_gpu=15, n_cpu=25 (from --layers 1/15/40)"])
        else:
            if not n or n < 2:
                continue
            p_all = (Tn - T1) / (n - 1)
            H = T1 - p_all
            w_all = (Wn - W1) / (n - 1)
            rows.append([model, fmt(p_all, 3), fmt(p_all, 3), fmt(H, 3),
                         fmt(w_all, 3), fmt(w_all, 3),
                         "all %d layers uniform (from --layers 1/%d)" % (n, n)])
    if rows:
        L.append(md_table(["model", "GPU-layer GPU ms", "CPU-layer GPU ms",
                           "head (norm+lm_head) GPU ms", "GPU-layer wall ms",
                           "CPU-layer wall ms", "note"], rows))

    # ---- dense models: CUDA-graph-mode wall differencing (true GPU time, no submission
    #      gaps, because the whole 32-layer pipeline is replayed as one graph) ----
    drows = []
    for model in model_order:
        modes = runs[model]
        gl1 = modes.get("graph_layers1")
        if not gl1:
            continue
        cand = [k for k in modes if k.startswith("graph_layers") and k != "graph_layers1"]
        if not cand:
            continue
        mode_n = cand[0]
        try:
            N = int(mode_n.replace("graph_layers", ""))
        except ValueError:
            continue
        gln = modes[mode_n]
        W1 = gl1["engine"].get("token_wall_median")
        WN = gln["engine"].get("token_wall_median")
        if not W1 or not WN or N < 2:
            continue
        p = (WN - W1) / (N - 1)
        H = W1 - p
        md = MODELS.get(model, {})
        ntot = md.get("n_layers")
        dtype = md.get("dp4a_bytes_mb")
        eff = None
        if ntot:
            total_gpu = p * ntot + H
            eff = total_gpu, (md.get("weights_gib", 0) * 1024.0 / total_gpu
                              if total_gpu > 0 else 0.0)
        drows.append([model, fmt(W1), fmt(WN), fmt(p, 4), fmt(H, 4),
                      fmt(eff[0]) if eff else "-", fmt(eff[1], 1) if eff else "-"])
    if drows:
        L.append("\n### dense models: true per-layer GPU time from CUDA-graph wall differencing\n")
        L.append("In CUDA-graph mode there are no host submission gaps, so W(1)=p+H and "
                 "W(N)=N*p+H; p is the real GPU time per layer and H the final "
                 "norm+lm_head.  The last two columns extrapolate the full-model GPU "
                 "time and the implied effective weight bandwidth.\n")
        L.append(md_table(["model", "W(1 layer) ms", "W(N) ms", "GPU ms/layer",
                           "head H ms", "full-model ms", "effective GB/s"], drows))

    # ---------------- per-layer decomposition ----------------
    for model in model_order:
        for mode in sorted(runs[model]):
            r = runs[model][mode]
            rows = r.get("layer_rows")
            if not rows:
                continue
            att = r.get("attribution") or {}
            L.append("\n## %s / %s: per-layer decomposition\n" % (model, mode))
            L.append("_kernel attribution: %d layers, %d MoE layers, %d MoE layers on GPU; "
                     "uniform kernel time %.2f ms/layer, GPU-layer-only %.2f ms/layer, "
                     "run-level %.2f ms; assigned %.2f of measured %.2f ms; ambiguous %d_\n"
                     % (att.get("n_layers", 0), att.get("n_moe", 0), att.get("n_gpu_moe", 0),
                        att.get("uniform_per_layer_ms", 0.0),
                        att.get("gpu_layer_extra_ms", 0.0), att.get("run_level_ms", 0.0),
                        att.get("assigned_total_ms", 0.0), att.get("kprof_total_ms", 0.0),
                        len(att.get("ambiguous") or [])))
            rows_md = []
            lu = r.get("layer_util_map") or {}
            for x in rows:
                rows_md.append([x["layer"], x["cls"], fmt(x["wall"]), fmt(x["gpu"]),
                                fmt(x["host"]), fmt(x["idle"]), fmt(x["sync_ms"]),
                                fmt(x.get("util"), 0)])
            L.append(md_table(["layer", "class", "wall ms", "GPU ms", "host matvec ms",
                               "idle ms (wall-gpu-host)", "sync ms", "NVML util%"], rows_md))
            cs = r.get("idle_totals") or {}
            if cs:
                L.append("\n_class totals: `%s`\n" % json.dumps(cs))

    # ---------------- kernels ----------------
    for model in model_order:
        for mode in sorted(runs[model]):
            r = runs[model][mode]
            kp = r.get("kprof")
            if not kp:
                continue
            L.append("\n## %s / %s: kprof kernel kinds (last decode token)\n" % (model, mode))
            rows = [[k["name"], fmt(k["ms"]), k["calls"], fmt(k["ms"] / k["calls"], 4),
                     fmt(k["pct"], 1)] for k in kp["kernels"]]
            L.append(md_table(["kernel", "total ms", "calls", "ms/call", "% GPU"], rows))

    # ---------------- text time series ----------------
    L.append("\n## text time series of the decode window\n")
    L.append("`▁`=0 %% .. `█`=100 %% NVML util, one char per 50 ms minimum bin; the second "
             "line is GPU power (W).\n")
    for model in model_order:
        for mode in sorted(runs[model]):
            se = runs[model][mode].get("series")
            if not se:
                continue
            L.append("\n### %s / %s (bin %.0f ms)\n" % (model, mode, se["bin_ms"]))
            L.append("util : " + sparkline(se["util"]) + "\n")
            p = se["power"]
            pmin = min([x for x in p if not math.isnan(x)]) if any(not math.isnan(x) for x in p) else 0
            pmax = max([x for x in p if not math.isnan(x)]) if any(not math.isnan(x) for x in p) else 1
            L.append("power: " + sparkline(p, pmin, max(pmax, pmin + 1)) + "\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="single run directory (or its .meta.json prefix parent)")
    ap.add_argument("--root", help="suite root: root/<model>/<mode>/")
    ap.add_argument("--json", help="write this JSON file")
    ap.add_argument("--md", help="write this markdown file")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not a.run and not a.root:
        ap.error("need --run or --root")
    if a.run:
        r = Run(a.run)
        rep = run_report(r)
        print(json.dumps(rep, indent=1))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1)
        return 0
    runs = analyse_suite(a.root)
    md = suite_markdown(runs)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=1)
    if not a.quiet:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
