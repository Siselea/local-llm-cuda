# ARCH-GENERALITY: metadata-driven architecture support

Status: 2026-11-22. Owner: generalize. Commits: `8099b6c` (generic substrate),
`d9bb37c` (extended dense path + gemma4), `dd96fa8` (legacy-path regression fix +
`arch_ext` default off), `5014cc6` (this report).  Verification on an isolated
worktree build of main+fixes: `tools/smoke_models.sh` 4/4 green and
`tools/gen_regression.sh` **10/10 PASS** (incl. coder_v2_lite 185).

## 1. Why this work exists

The user's criticism was that the engine only worked for the model families it
had been explicitly adapted to: *"你只对你测试过的模型有适配，其余没测试过的一律没有
适配，严重丧失了通用性"*.  This round attacks that at the loader/IR level: the
engine now derives per-layer attention geometry and the optional architecture
ops from **GGUF metadata and tensor structure only** — no architecture-name
special cases — and executes them through a generic dense layer path.

Two things are deliberately separated:

1. **Generic substrate** (required by any future architecture):
   - GGUF parser keeps small numeric metadata arrays (`attention.head_count_kv`,
     `attention.sliding_window_pattern`, ...).  This is cheap: arrays are only
     retained when numeric and <= 4096 elements, weight payloads are still
     skipped.
   - IR builds `LayerAttn` per layer: `n_kv`, `key_dim`, `value_dim`,
     `rope_dim`, `rope_base`, `window`, `has_v` (attention_k_eq_v),
     `qk_norm`, `v_norm`, `sandwich`, `layer_scalar`.
     Derivation rule: scalar metadata is the stack-wide default; array metadata
     and the `*_swa` variants select per-layer values; optional tensors
     (`attn_q_norm`, `post_attention_norm`, `layer_output_scale`, ...) enable the
     matching op.
   - `ArchCaps` records which optional ops the model asks for (qk-norm,
     v-norm, sandwich norms, layer output scalar, GELU-tanh MLP, embedding
     scale, attention scale, final logit softcap).
   - Four new composable kernels: `gelu_mul`, `attn_cached_cfg` (configurable
     logit scale + sliding window), `scale_inplace`, `softcap_inplace`.
   - Config switches (`arch_ext`, `arch_act`, `arch_embed_scale`,
     `arch_attn_scale`, `arch_norm_mode`, `arch_softcap`) so every new behaviour
     can be vetoed at runtime and investigated per model.
2. **The generic extended dense path** (loader + forward) that consumes the
   above.  It is only taken when the metadata genuinely needs it **and** the
   model is a pure-attention dense model:
   - `caps.extended` = per-layer attention geometry OR sandwich norms OR layer
     output scalar OR final logit softcap;
   - hard safety rails: never for hybrid-SSM (qwen35), MLA (deepseek2/3) or MoE
     models — those keep their dedicated, verified paths;
   - `arch_ext=off` restores the legacy path completely.

## 2. Support matrix (local models, 2026-11-22)

| model | architecture | loads | generates | numerical parity vs llama.cpp |
|---|---|---|---|---|
| qwen2.5-0.5b-instruct Q4_K_M | llama (dense GQA) | yes | yes | yes (smoke top-1 40716, gen_regression) |
| DeepSeek-Coder-V2-Lite Q5_K_M | deepseek_v2 (MLA + MoE) | yes | yes | yes in smoke (top-1 185) *** |
| Qwen3.5-9B Q5_K_M | qwen35 (hybrid SSM) | yes | yes | yes (smoke top-1 2614) |
| Qwen3.6-35B-A3B Q4_K_M | qwen35moe (hybrid SSM + MoE) | yes | yes | yes (smoke top-1 2614) |
| **gemma-4-12b-it Q4_K_M** | gemma4 dense, mixed layer types | **yes** | **yes** | **NO** — 8-token greedy diverges at token 3; max Δlogp over llama top-5 = 0.46 (top-2 near-tie flip). Requires `--arch-ext auto` (default off until parity is proven). |
| Gemma4-26B-A4B QAT Q4_K_M | gemma4 MoE (fused `ffn_gate_up_exps`) | no | no | not implemented; clear diagnostic (needs fused-expert gate/up slab) |
| gpt-oss-20b MXFP4 | gpt-oss | no | no | parse error: unsupported GGUF tensor type id 39 (MXFP4) — tracked as Phase 3 |
| Qwen3.8-27B / R1-Distill-14B | qwen2 family | (Phase 2 pending) | — | — |

*** Coder-V2-Lite top-1 is stable, but a full `gen_regression` run reported the
coder case as FAIL in one run while the shared tree was being rebuilt by
multiple agents; treat that as environment noise until re-run on an idle tree.

### Regression fix (found by an independent full-suite run)

`d9bb37c` briefly regressed MLA first-token output (DeepSeek-Coder-V2-Lite:
expected 185, got 75723).  Root cause was not the MLA code path (the hard rails
kept it on its own loader), but the final-norm upload: the generic work replaced
`upload_norm()` with a plain upload for every model, bypassing upload_norm's
`deepseek2` zero-centering convention.  `dd96fa8` restores `upload_norm()` for
all legacy paths (the plain upload now only serves the explicit
`arch_norm_mode=plus1` override) and also stops the IR from rewriting the
stack-wide key/value lengths unless the GGUF really carries per-layer geometry.
Verified after the fix: smoke 4/4, gen_regression 10/10.

### gemma4-12b parity detail

The real checkpoint now loads through the generic path and produces plausible
tokens, but is **not numerically aligned** with llama.cpp yet:

- Greedy prompt ids `[9259]` ("Hello"): llama.cpp → `236770 236770 ...` (8x);
  engine → `236761 236761 236761 236770 236770 236770 236770 236761`.
  Both tokens are llama.cpp's top-2 (`236770` logp -0.645, `236761` -0.985);
  the engine prefers the second one, i.e. a systematic logit error of ~0.3–0.5.
- A **tiny synthetic gemma4 GGUF** (same metadata structure/tensor layout, 64
  hidden, 2 layers, one sliding + one full layer, tied head, softcap 30) is
  matched by the same math to **relL2 6.7e-5** on the final hidden state and
  **1.4e-6** on top-k log-probs (llama-server `/embedding` with token ids as
  `content`, so no BOS).  Variants confirm: direct RMSNorm weights (not 1+w),
  sqrt(hidden) embedding scale, GELU-tanh MLP, weightless V norm,
  k_eq_v = weightless RMSNorm(raw K projection), sandwich norms after the
  projections, final layer scalar, attention scale 1.0, repeat_interleave GQA
  (h/rep) mapping, `window` = attend current + previous `window-1`.
- Additional evidence that the remaining gap is *quantized-backend numerics*,
  not layer semantics: (a) llama.cpp itself disagrees between CPU and GPU on the
  real checkpoint (top-2 log-prob gap -0.63 on CPU vs -0.34 on GPU; final hidden
  norm 220.2 vs 213.6), while the engine lands between them; (b) on a truncated
  8-layer copy of the real checkpoint our f32 reference deviates from llama.cpp
  CPU by relL2 7.6e-2 already after 8 layers (F32 tiny model: 6.7e-5), which is
  the expected footprint of llama.cpp's runtime activation quantization
  accumulating with depth.  Exact greedy parity with llama.cpp on a Q4_K
  checkpoint is therefore backend-numerics-bound at near-ties (top-2 separation
  ~0.3 logit), not a missing architecture rule.
- Therefore the layer *math* is right; the real-checkpoint difference is in
  something only the real file has.  Ruled out so far: architecture-name
  heuristics, activation, embedding scale, sandwich/scalar placement, V
  construction.  For a 1-token prompt rotary is a no-op (position 0), so
  `rope_freqs` cannot explain the first-token difference; however it still must
  be implemented for multi-token parity.  Prime suspect list is maintained in
  §4.

## 3. Generic path implementation notes

- Loader (`load_ext_layer`): loads projections as quantized-resident tensors
  (`load_qtensor`) with an f32 fallback, per-layer norms (optional `1+w` mode),
  per-head qk-norm weights, a ones vector for weightless V norm, per-layer rope
  frequencies from the derived base/dim, a per-layer KV cache sized by that
  layer's `n_kv x max_seq x head_dim`, and the layer output scalar.
- Forward (`run_layer_ext`): pre-norm → q/k/v projections → (k_eq_v: V =
  weightless RMSNorm(raw K) before k-norm/rope) → qk-norm → rope (per-layer
  freqs/base, `rope_strided` kernel) → KV write → `attn_cached_cfg` with the
  derived scale/window → output projection → sandwich norm → residual →
  MLP (GELU-tanh or SiLU) with sandwich norm → residual → layer scalar.
- Embedding multiplier and final logit softcap are applied in the shared
  forward; the tied LM head reuses `token_embd.weight`.
- CUDA graphs are disabled for the extended path (per-layer geometry differs;
  keeping the plain launch path avoids baking shapes into a graph).
- Validation: per-layer checks (n_kv divides n_head, even head/rotary dims,
  rotary dim <= head dim, positive rope base, equal K/V dims for the shared
  KV-write kernel) with actionable messages; an informational warning lists
  which capabilities the model enabled.

## 4. Known gaps / next steps

1. **gemma4-12b numerical parity** (highest priority follow-up):
   - implement proportional rotary via `rope_freqs.weight` as frequency
     factors (`theta_i /= factor_i`; 1e30 marks "no rotation"), verified by
     multi-token prompts;
   - layer-wise bisect on a truncated real checkpoint (identity-zeroed later
     layers) to localise the first divergent layer (tooling added:
     `tools/trunc_bisect.py`);
   - compare against llama.cpp CPU vs GPU numerics to separate architecture
     semantics from backend rounding.
2. **gemma4-26B MoE**: fused `ffn_gate_up_exps` slab + `.scale` tensors,
   expert routing for gemma4, and the extended path has to grow a MoE variant
   (currently excluded by the safety rails).
3. **gpt-oss-20b (Phase 3, report only)**: GGUF type 39 (MXFP4) parsing,
   sliding-window attention, YaRN-style rope, MoE layout; feasibility/cost
   analysis still owed.
4. **Phase 2**: verify Qwen3.8-27B and R1-Distill-14B qwen2-family checkpoints
   load + match llama.cpp greedy prefixes; they exercise scalar metadata only,
   so they should pass through the legacy path untouched.
5. Loader robustness follow-ups from the earlier round (nested ARRAY depth
   bound, tensor-offset overflow detection, clearer unsupported-architecture
   diagnostics) are still uncommitted work; they are independent of this layer.

## 5. Performance / cost

- gemma4-12b Q4_K_M: 6777 MiB quantized weights + 192 MiB KV at ctx 128 +
  activations; on the 12 GB RTX 3060 the auto placement ends up with ~0 MiB
  spare at ctx 128 (driver paging, ~5-18 tok/s).  For interactive use ctx must
  stay small or a larger-margin placement is needed; this is a property of the
  12B model, not of the generic path.
- Tiny gemma4 fixture (281 KB): loads in ms, used as a millisecond-level
  semantics oracle against llama.cpp.
- Generic path overhead: one extra launch per op vs the fused legacy blocks
  (qk-norm, V norm, scale, softcap are separate tiny kernels); irrelevant at
  these sizes and only active for extended models.

## 6. Reproduce

```bash
# tiny oracle + parity anchor (needs llama-server; CPU is enough)
python tools/make_tiny_gemma4.py
U:/llama/llama-server.exe -m build/fixtures/tiny_gemma4/model.gguf --ctx-size 128 \
    -ngl 0 --port 8098 --temp 0 --top-k 1 --parallel 1 --embeddings &
python tools/tiny_gemma4_grid.py --ids 5,7,9
# real checkpoint (GPU): load + generate
./build/test_gen.exe "U:/AI/gemma-4-12b-it-Q4_K_M.gguf" 9259 8 --ctx 64 --arch-ext auto
```
