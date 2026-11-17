# MTP / 投机解码 可行性报告（MTP-RECON）

> 只读侦察 + 方案；本文件是唯一新增/修改的仓库文件。
> 侦察时间：2026-11-15 07:40–08:20（本机 RTX 3060 12GB / i5-10400F / 32GB DDR4）。
> 所有实测命令都在仓库外执行（`%TEMP%\mtprecon\` 下的 `build/engine.exe` 副本），未触碰 `src/`、`include/`、`build.sh`、`tests/`、`tools/`（另一个智能体正在改 `src/main.cpp`，本报告所有结论不依赖它的改动）。

---

## 0. TL;DR（先看这 6 条）

1. **四个模型里只有 Qwen3.6-35B-A3B 带 MTP 头**（`qwen35moe.nextn_predict_layers=1`，MTP 层 = `blk.40`，20 个张量）。Qwen3.5-9B（qwen35）、Coder-V2-Lite（deepseek2）、qwen2.5-0.5b（qwen2）**都没有**。
2. **额外发现：`<MODEL_DIR>/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` 也带 MTP 头**（`qwen35`，MTP 层 = `blk.32`）。它就是 llama.cpp 上 `+29.5%` 实测用的那个模型（`<MODEL_DIR>` 下有同名同大小 5,887,667,808 B 的副本），**引擎现在就能加载它（VALIDATION PASS，只是把 MTP 层当 unused tensor 丢掉）**，实测 decode 32.4 ms/token（31 tok/s）。
3. 引擎对 MTP 的支持**只到"检测并丢弃"**：`src/ir.cpp:226-239` 从 `.nextn.` 张量名推断 `mtp_layers`、把 `n_layer` 钳到 40/32、把这 20/15 个张量标成 `Role::OTHER`，然后 `load_weights()` 根本不加载它们。**没有任何 draft / verify / rollback 代码**（全仓 grep `nextn|mtp|spec_type|draft` 只有 `src/ir.cpp` 里同一处 if 块的 10 行 + `src/main.cpp` 的 1 行 IR 输出）。
4. **最大阻塞不是 MTP，而是引擎的 n_tokens>1 前向路径**。实测：9B 单 token 前向 22–23 ms，而 n=2/3/4/8/32 的批前向**恒定 326–357 ms**（与 batch 大小无关）。原因是 `matmul()` 在 `n_tokens>1` 时逐块把整模型反量化成 f32（`src/runner.cpp:2867-2930`），且 `lm_head` 在 rows>1 时走 f32 物化（`src/runner.cpp:1530-1534`，9B 的 f32 头 = 4.07 GB、35B = 2.03 GB）。**这条路径直接做 verify = 必亏（13–15× 单步成本）**；`--quant-gemm`（实测 568–570 ms）和 `--half-t`（实测 4.8 s）都救不了。
5. 因此 P0 的唯一硬门槛是 **2–4 行量化 GEMV（读一遍权重算多行）+ 多行 lm_head**，把 "3 token verify" 压到 ≈1.2× 单步 decode。有了它：
   - **Qwythos-9B-MTP（全 GPU、无 CPU 专家）预期 31 → 41–48 tok/s（+34% ~ +55%）**；
   - **Qwen3.6-35B-A3B 预期 +0% ~ +7%**（验证批次被 CPU MoE 拖到 ≈1.8×，且显存只剩 1137 MiB、MTP 层 545 MB 必须上 CPU）；
   - Coder-V2-Lite / qwen2.5-0.5b / Qwen3.5-9B 没有 MTP 头，本轮不做（外部草稿模型那条路用户已实测只 +7% 且未生效，见 `local-llm-engine-p0` 记忆）。
6. **如果只做一件事**：先做 "2–4 行量化 GEMV + 多行 head"（P0-A），在 Qwen3.5-9B 上量出 `T(3 token verify) / T(1 token decode)`。**≤1.25 → 立刻上 Qwythos-9B 的 MTP；≥1.5 → MTP 在本引擎上不值得做**，把这套 kernel 当作小 batch prefill 加速收掉（它同时修掉短 prompt 前向那 0.33 s 固定成本）。

---

## 1. 四个模型 + 2 个额外模型的 MTP 头盘点（用数据说话）

### 1.1 解析方法与数据来源

- 自写只读 GGUF 探针（仓库外，`%TEMP%\mtprecon\gguf_probe.py`）：读 header + KV + tensor info，**不读张量数据**，把 `meta` / `tensors(name,dims,type)` 落成 JSON。
- 复用仓库既有的 `build/q35b_meta.json`、`build/coder_meta.json` 做交叉验证（张量清单一致）。
- 引擎侧证据：`./build/engine.exe info <model>`（只解析元数据，不加载权重）。

### 1.2 总表

| 模型文件 | arch | block_count | `nextn_predict_layers` | MTP 层 | MTP 张量数 | 结论 |
|---|---|---|---|---|---|---|
| `Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf` | `qwen35moe` | **41**（40 主 + 1 MTP） | **1** | `blk.40` | **4 个 nextn + 16 个层张量 = 20** | ✅ **有** |
| `Qwen3.5-9B-Q5_K_M.gguf` | `qwen35` | 32 | — | — | 0 | ❌ 无 |
| `DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf` | `deepseek2` | 27 | — | — | 0 | ❌ 无 |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | `qwen2` | 24 | — | — | 0 | ❌ 无 |
| `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf`（额外） | `qwen35` | **33**（32 + 1） | **1** | `blk.32` | **4 + 11 = 15** | ✅ **有** |

判据（四条同时成立才算"有 MTP 头"）：
1. 元数据键 `*.nextn_predict_layers ≥ 1`；
2. `block_count == 主层数 + nextn_predict_layers`；
3. 存在 `blk.{N}.nextn.*` 张量（N = block_count-1）；
4. 该 block 同时含完整的一层 attention/FFN 张量（不是只有投影）。

### 1.3 Qwen3.6-35B-A3B：`blk.40` 完整张量清单（这就是 MTP 层）

```
blk.40.attn_norm.weight              [2048]            F32
blk.40.attn_q.weight                 [2048, 8192]      Q4_K    # gated q: 2*H*D = 2*16*256
blk.40.attn_k.weight                 [2048, 512]       Q4_K    # 2 个 kv head * 256
blk.40.attn_v.weight                 [2048, 512]       Q6_K
blk.40.attn_q_norm.weight            [256]             F32     # per-head QK RMSNorm
blk.40.attn_k_norm.weight            [256]             F32
blk.40.attn_output.weight            [4096, 2048]      Q4_K    # 16*256 -> hidden
blk.40.post_attention_norm.weight    [2048]            F32
blk.40.ffn_gate_inp.weight           [2048, 256]       F16     # router
blk.40.ffn_gate_inp_shexp.weight     [2048]            F16     # shared-expert gate
blk.40.ffn_gate_shexp.weight         [2048, 512]       Q4_K
blk.40.ffn_up_shexp.weight           [2048, 512]       Q4_K
blk.40.ffn_down_shexp.weight         [512, 2048]       Q6_K
blk.40.ffn_gate_exps.weight          [2048, 512, 256]  Q4_K    # 256 专家
blk.40.ffn_up_exps.weight            [2048, 512, 256]  Q4_K
blk.40.ffn_down_exps.weight          [512, 2048, 256]  Q6_K
blk.40.nextn.eh_proj.weight          [4096, 2048]      Q4_K    # 2*hidden -> hidden
blk.40.nextn.enorm.weight            [2048]            F32     # emb 的 RMSNorm
blk.40.nextn.hnorm.weight            [2048]            F32     # hidden 的 RMSNorm
blk.40.nextn.shared_head_norm.weight [2048]            F32     # 出 head 前的 RMSNorm
```

对照 `blk.39`（主模型最后一个真层）：**16 个张量，与 blk.40 的 16 个逐一对齐**（`ffn_gate_inp`/`ffn_gate_inp_shexp` 在 blk.40 是 F16、blk.39 是 F32，其余同名同形）。
结论：**MTP 层不是"一个投影+head"，而是一个完整的 attention + MoE 层**（qwen3-next 风格），前面多一个 `eh_proj` 融合投影、一个 `shared_head_norm` 出 head 前归一。

### 1.4 Qwythos-9B（qwen35 稠密版）：`blk.32` 清单

```
blk.32.attn_{q,k,v,output}.weight        Q8_0  (4096x8192 / 4096x1024 / 4096x1024 / 4096x4096)
blk.32.attn_{q,k}_norm.weight            F32 [256]
blk.32.attn_norm / post_attention_norm   F32 [4096]
blk.32.ffn_{gate,up,down}.weight         Q8_0  (4096x12288 / 4096x12288 / 12288x4096)
blk.32.nextn.eh_proj.weight              [8192, 4096]   Q8_0
blk.32.nextn.enorm.weight                [4096]         F32
blk.32.nextn.hnorm.weight                [4096]         F32
blk.32.nextn.shared_head_norm.weight     [4096]         F32
```

同一套结构（完整 attention + FFN 层 + 4 个 nextn 张量），只是稠密 FFN、hidden=4096。

### 1.5 共同的形状结论（决定实现难度）

| 事实 | 35B-A3B | Qwythos-9B | 含义 |
|---|---|---|---|
| `eh_proj` 形状 | `[2*hidden, hidden]` = `[4096, 2048]` | `[8192, 4096]` | 输入 = `concat(enorm(emb(t)), hnorm(h))`，维数是 **2×hidden**，实锤"拼 embedding 与 hidden" |
| `enorm` / `hnorm` | 各 `[hidden]` | 各 `[hidden]` | 两个独立 RMSNorm，分别归一 embedding 和 hidden（不是共用一个） |
| `shared_head_norm` | `[hidden]` F32 | `[hidden]` F32 | 出 head 前再归一 |
| `shared_head_head` | **不存在** | **不存在** | head 与主模型 **共用 `output.weight`**（两模型都是 untied embedding：`token_embd`≠`output.weight`） |
| `nextn.embed_tokens` | **不存在** | **不存在** | 草稿 token 的 embedding 直接复用 `token_embd.weight` |
| MTP 层里有无 SSM 张量 | **无**（纯 attn+MoE） | **无**（纯 attn+FFN） | ⭐ **MTP 层永远是 full-attention 层**，与主模型的 `full_attention_interval=4` 无关；它的 KV cache 是位置索引的，回滚免费 |
| MTP 层字节数 | ≈ **545 MB**（专家 522 MB 占 96%） | ≈ **258 MB** | 35B 的 MTP 层在 12GB 卡上必须整层放 CPU；9B 可以进 VRAM |
| MTP 层 KV/token | 2 kv head × 256 × 2 × 4B = 4 KiB | 4 × 256 × 2 × 4B = 8 KiB | 8K ctx 只需 16–33 MB |

### 1.6 引擎现在到底读到/支持到什么程度

| 环节 | 现状 | 证据 |
|---|---|---|
| 发现 MTP 层 | ✅ 有（靠 `.nextn.` 张量名，不读 `nextn_predict_layers` 元数据键） | `src/ir.cpp:226-239` |
| 计数 | ✅ `m.mtp_layers = 1`，IR JSON 里有 `mtp_layers` 字段 | `include/engine/ir.h:86`，`src/main.cpp` 的 ir 输出 |
| 层数钳制 | ✅ `n_layer = min(n_layer, 40)`，MTP 层的张量全部 `Role::OTHER` | `src/ir.cpp:234-236` |
| 权重加载 | ❌ **从不加载**（`load_weights()` 只循环 `0..n_layer-1`） | `src/runner.cpp:719`、`:781` |
| 显存/预算 | ✅ 被排除（`fixed_bytes()` 只统计 EMBED/OUTPUT/ATTN/... ） | `src/ir.cpp:91-100` |
| draft/verify/回滚 | ❌ **不存在**（全仓无 spec/accept/rollback 代码） | grep 结果 |

`engine.exe info` 的实机输出（35B）：

```
layers     : 40   hidden=2048  ffn=0  vocab=248320
roles      : attn:370 embed:1 expert:120 norm:41 other:20 output:1 router:40 shared_expert:160
VALIDATION : PASS
warn       : MTP head detected on blocks; excluded from 40 layers
```

→ **20 个 `other` 就是被丢弃的 MTP 层**（4 个 nextn + 16 个层张量）。Qwythos-9B 同理（`other:15`）。

> 小缺口：GGUF 里明明有 `qwen35moe.nextn_predict_layers = 1` / `qwen35.nextn_predict_layers = 1`，引擎没读，纯靠张量名推断。建议 P0 顺带读元数据键做交叉校验（更稳，且能给 `info` 打印真实训练粒度）。

---

## 2. Qwen3.6 的 MTP 具体结构（从张量反推）+ llama.cpp 参照

### 2.1 反推出的计算图（与 DeepSeek-V3 系 MTP 一致）

主模型第 40 层之后的残差流 hidden 记 `h_p`（位置 p），主模型 head 采样出 `t_{p+1}`：

```
draft 输入:  u = eh_proj( concat( enorm(emb(t_{p+1})), hnorm(h_p) ) )     # [2H] -> [H]
MTP 层:      同主模型一层，但恒为 full-attention：
             a = u + attn( rmsnorm_attn(u) )              # 自己独立的 K/V cache，位置 p+1
             y = a + ffn_or_moe( rmsnorm_post(a) )        # 35B 是 MoE（256 选 8 + shared）
draft logits: z = lm_head( shared_head_norm(y) )           # 复用 output.weight
draft token:  d = argmax(z)                                # 训练目标 = t_{p+2}
```

链式多步 draft：把 `y` 当作新的 hidden，`d` 当作新的输入 token，重复上面两步，第 i 步在位置 `p+1+i` 写 MTP 层的 KV。

**关键点（与常见教程不同，务必按此实现）**：
- 只有 **1 个** MTP 层（`nextn_predict_layers=1`），链式 draft 是"重复用同一层 + 每步写一次 KV"，不是多层堆叠；
- MTP 层的 attention 恒为 full-attention（张量清单里没有 `ssm_*`），**与 `full_attention_interval=4` 无关**——35B 主模型 40 层里 30 层是 SSM，但 MTP 层不是。反证：`blk.40` 的 `(40+1) % 4 == 1`，按主模型规则该是 "linear"（SSM）层，但它实际是 attention 层（`attn_q/k/v/output` + `attn_q_norm/k_norm`），说明 MTP 层走的是固定的"整层 attention"图；
- `eh_proj` 的输入维是 **2×hidden**，顺序是 `[enorm(emb) 在前, hnorm(h) 在后]`（顺序决定数值正确性，**未验证**，需用 llama.cpp 对拍或接受率验证）。

### 2.2 llama.cpp b12128/b11718 的实现线索（只读二进制字符串 + 用户实测日志）

`U:\llama` 只有二进制没有源码（`strings` 可读出符号与日志串，说明未 strip）。可**确认**的事实：

| 事实 | 证据（`strings llama.dll` / `llama-common.dll`） |
|---|---|
| 每个 arch 有独立 MTP 计算图 | `.?AUgraph_mtp@llama_model_qwen35@@`、`@qwen35moe`、`@qwen3next`、`@deepseek2/32/4`、`@glm4_moe`、`@glm5_next`、`@cohere2moe`、`@bailingmoe3`、`@nemotron_h_moe`、`@qwen4exp`… |
| qwen35 MTP 需要 `nextn.shared_head_norm`，缺失时回退 `output_norm` | `"QWEN35 MTP: missing both nextn.shared_head_norm and output_norm"` |
| 期望的张量名与本机 GGUF **一致** | `blk.%d.nextn.{eh_proj,enorm,hnorm,shared_head_norm,embed_tokens,shared_head_head,hc_head_*}` |
| 目标模型要"导出"nextn 隐状态给草稿侧 | `llama_get_embeddings_nextn`、`llama_set_embeddings_nextn`、`llama_set_nextn_layer_offset`、`"MTP input row width must match the target h_nextn width"` |
| 草稿侧是**独立的 draft context**（自己的 KV cache / 自己的放置参数） | `"creating MTP draft context against the target model"`、`--cache-type-k-draft`、`--gpu-layers-draft`、`--cpu-moe-draft`、`--device-draft`、`--spec-draft-ngl`、`--spec-draft-ncmoe`、`--spec-draft-n-ubatch` |
| 草稿实现有状态保存/恢复接口 | `common_speculative_impl_draft_mtp::{begin,draft,accept,process,get_state,set_state}` |
| CLI 超参 | `--spec-type draft-mtp`、`--spec-draft-n-max`、`--spec-draft-n-min`、`--spec-draft-p-min`、`--spec-draft-p-split`、`--spec-draft-prio{,-batch}`、`--spec-draft-poll{,-batch}` |
| p-min 是"置信度阈值"（MTP 无 confidence head 时要求置 0） | `"DSpark draft has no confidence head: please set --spec-draft-p-min 0"` |
| **hybrid/SSM 模型的部分回滚是显式机制**，不是所有模型都支持 | `"n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0"`、`"disabling rollback snapshots because the memory module is empty"`、`"seq %d pos [%d, %d] rollback=%u"`、`"DSV4 recurrent state rollback index out of range"` |

> 结论：llama.cpp 的方案 = **目标 context 多算一个 h_nextn → 独立草稿 context 跑 MTP 层图（自己的 KV）→ 目标 context 做 k+1 token 的批验证 → recurrent 状态用"回滚快照"恢复**。这三件事我们都要自己做。

**用户实测里可直接引用的超参/收益**（`<MODEL_DIR>\AI-最优启动\实测原始数据.txt`）：

| 模型 / build | 无 MTP | 有 MTP | n-max | 接受率 α | 备注 |
|---|---|---|---|---|---|
| Qwythos-9B-MTP 192K（2026-11-04 b12110） | 50.26 | 62.79 → **63.33**（greedy）→ 63.67（p-min） | 2 | 0.5645 | n-max=2 优于 n-max=3（56.82, α=0.4449）与 1（63.40 略低） |
| Qwythos-9B-MTP 192K（b11718, q4v） | — | **72.09** | 2 | 0.7284 | q4 KV + n-max=2 是该项最好 |
| Qwythos-9B-MTP 64K（b11156 旧） | 46.4 | 49.7 | 默认 | — | 用户最初引用的 `+7.1%` 就是这一组 |
| Qwythos-9B-MTP（b12128，2026-11-12 单用户） | 42.23 | **54.70（+29.5%）** | 默认 3 | 0.484 | **用户记忆里"+29.5%"的出处**；mean draft len 2.44 |
| **Huihui-35B-A3B 64K（b11718, ncmoe27, t6）** | **26.16** | **27.89（+6.6%）** | 默认 | 0.6157 | 同 build 同 prompt 的干净对照 |
| Huihui-35B-A3B 64K（b11718） | — | 24.12(n3) / **25.76**(n2) / 26.43(n2,temp0.6) / 20.36(n4) | 2 最优 | 0.45 / 0.505 / 0.55 / 0.337 | n-max=4 是负优化 |
| Huihui-35B-A3B（b12110） | — | 26.93 (n2) / 22.94 (n3) | 2 | — | |
| nerkyor-27B 稠密 qwen35（b11718） | 2.26 | **2.92（+29%）** | 2 | 0.79 | 稠密模型 MTP 收益明显更大 |

**这张表是本报告收益模型的校准基准**：稠密全 GPU 模型 +29%，35B-A3B（CPU 专家 27 层）只有 +6.6%。

---

## 3. 实现方案（具体到能直接开工）

### 3.1 装载：把 MTP 层当"第 N+1 层"单独装

| 改动点 | 位置 | 内容 |
|---|---|---|
| IR 读元数据键 | `src/ir.cpp:226` 附近 | 优先读 `{arch}.nextn_predict_layers`；缺失时退回 `.nextn.` 张量名推断；额外给 `ir::Model` 加 `mtp_block`（MTP 层 block 号 = `nextn_predict_layers>0 ? n_layer_real : -1`） |
| 张量角色 | `src/ir.cpp:235-236` | 现在把 MTP 层全部标 `OTHER`。改成标"属于 MTP 层"（建议新枚举 `Role::MTP` 或保留现有 role 但记录 `mtp_index`），避免 `validate()` 误判 |
| 权重加载 | `src/runner.cpp:781` 之后 | 新增 `load_mtp_block()`：复用 `load_q35_attn(mtp_block, mtp_layer)` + `load_q35_ffn(mtp_block, mtp_layer)`，`mtp_layer.kc/vc` 独立分配（4–8 KiB/token）；`nextn.*` 四个张量单独读（`eh_proj` 走 `load_qtensor`，三个 norm 走 `tensor_f32`）。⚠️ **坑**：`load_q35_ffn(i, w)` 内部读 `ir_.layers[i].n_expert_tensors`（`src/qwen35.cpp:61`），而 `build_from_gguf` 在填充 `layers` 之前就把 `n_layer` 钳到 40 了（`src/ir.cpp:234` vs `:241`）⇒ 索引 40 越界。必须先在 ir 里为 MTP block 留一条 `LayerDesc`（把 `layers` 尺寸做成 `n_layer+1`，或单独存 `mtp_layer_desc`），否则 MoE 分支不会生效。 |
| 显存 | `src/planner.cpp` / `vram_*` | MTP 层计入预算并**优先放 GPU**；放不下则整层 CPU（35B 必然如此）。35B 当前 post-load free 只有 **1137 MiB < margin 1200**，装 MTP 层必须把 `gpu_moe_layers` 16→14，**decode 会掉 ~5%** —— 这一条单独就足以否掉 35B |

### 3.2 draft 路径（只跑 n=1，不需要新 kernel）

```
1) 主模型 forward(t_cur) 结束时，把最后一行残差流 hidden 拷进 h_last（[hidden] float）
   —— 位置：src/runner.cpp:1506 附近，run_pipeline 的 final-norm 之前；只在 spec 模式开启时做
2) draft_step(h_prev, tok, pos):
     e  = embed(tok)                       // 复用 embed_raw_ 的 CPU 反量化，已有
     u  = eh_proj( concat(enorm(e), hnorm(h_prev)) )   // matmul(n=1) 走现有 GEMV
     u  = mtp_attn_layer(u, pos)           // 复用 run_q35_attn_attn 的代码路径，独立 kc/vc
     y  = mtp_ffn(u)                       // 稠密→run_ffn；35B→run_moe（CPU 专家）
     logits = matmul(lm_head_qt_, shared_head_norm(y)) // 复用现有 head_topk 路径（只取 top-1）
     return argmax, y
3) 链式 k 步：h_prev ← y, tok ← argmax
```

成本（估算，按本机实测带宽 264 GB/s 折算）：

| 项 | 9B（Qwythos） | 35B |
|---|---|---|
| MTP 层权重 | 258 MB → **0.98 ms** | 545 MB，其中 CPU MoE 8 专家 ≈25 MB → **1.3 ms** + attn 0.1 |
| head（共用；9B/Qwythos Q6_K 1.017G 权重 = 834 MB；35B Q8_0 508.5M 权重 = 540 MB） | 834 MB → **3.2 ms** | 540 MB → **2.05 ms** |
| eh_proj + norm | 36 MB → 0.14 ms | 5 MB → 0.02 ms |
| **单步 draft 合计** | **≈ 4.3 ms（= 0.13 × decode）** | **≈ 3.4 ms（= 0.059 × decode）** |

> draft 是纯 n=1，**现有 GEMV/CUDA Graph 全部可用**，P0 不需要新 kernel。

### 3.3 验证路径（这里必须新增 kernel）

**目标**：把 `[t_{p+1}, d_1, ..., d_k]`（1+k 个 token）一次前向，拿到 k+1 行的 logits，逐行比较接受最长前缀。

现状实测（这就是必须解决的阻塞）：

| 前向 | Qwen3.5-9B | 说明 |
|---|---|---|
| n=1（decode，CUDA Graph） | **22.2 – 23.1 ms** | 正常 |
| n=2 / n=3 / n=4 / n=8 / n=32 | **347 / 326 / 339 / 357 / 355 ms** | **与 batch 大小无关 ⇒ 固定成本** |
| n=3 + `--quant-gemm` | 568 ms | 实验 GEMM 更慢 |
| n=3 + `--half-t` | 4772 ms | fp16 转置镜像，灾难 |

根因（读码 + 实测一致）：
1. `matmul()` 在 `n_tokens>1` 且没有 f32 镜像时，**逐块把整模型反量化到 f32 再 cuBLAS**（`src/runner.cpp:2867-2930`）；9B 权重 6.13 GB → f32 写读 ≈ 30 GB 流量 ≈ 330 ms。`prefill_f32_`（f32 常驻镜像）在 9B 上因 `dense_bytes*4 + 1GB > free` 被自动关掉（`src/runner.cpp:313-322`）。
2. `lm_head` 在 `rows>1` 时**必须物化 f32 头**（`src/runner.cpp:1530-1534`）：9B = 248320×4096×4 = **4.07 GB**，35B = **2.03 GB**。
3. `n_tokens>1` 不走 CUDA Graph（`src/runner.cpp:1610-1644` 只有 `n==1` 捕获），WDDM 提交开销按 FINAL-REPORT 记录可占 decode 墙钟 20–30%。

**P0-A 新增（核心工作量）**：

| 改动 | 文件 | 内容 | 预估 |
|---|---|---|---|
| 多行量化 GEMV | `include/engine/kernels.h`（新增 `R"cuda(...)"` 字符串） | `gemv_multi_{q4k,q5k,q6k,q8_0}(W, X[m][K], Y[m][N], m≤4)`：读一遍权重块、m 组累加器 → 权重流量不变、算力 ×m（dp4a 算力远未饱和）。**内层循环直接照抄现有 1 行 kernel**，保证数值一致 | ~250 行 CUDA 源串 + 8 行注册 |
| 派发 | `src/runner.cpp:2817` 前插分支 | `n_tokens<=4` → `gemv_multi_*`；`n_tokens>4` 仍走 prefill 老路 | ~40 行 |
| 多行 head | `src/runner.cpp:1523` | `rows<=4` 时用 `matmul(lm_head_qt_, rows, ...)`（多行 kernel）替代 f32 头；**禁止**再物化 `w_lm_` | ~15 行 |
| 每行 argmax | `include/engine/kernels.h`（复用 `k_topkpart_/k_topkfinal_`） | 加 `row` 偏移参数，或直接下载 m×vocab（4 行 × 248320 × 4B = 4 MB，**不推荐**，WDDM 上 ~0.35 ms）→ 建议 device 端每行 top-1，只拷 4 个 int | ~20 行 |
| 通用入口 | `include/engine/runner.h` | `std::vector<float> forward_rows(tokens, start_pos, int logit_rows, std::vector<int>* argmax_rows)`：保留每行的 argmax（不需要完整 logits 时） | ~30 行 |

**预期效果**：3 token verify ≈ 读一遍权重（9B 22.2 ms @264GB/s）+ attention/activation 增量（3 个位置 ≈ +1.5 ms）+ 启动开销（无 graph 时 +20–30%）。**若同时把 verify 也做成 CUDA Graph（P1），c ≈ 1.10–1.15；不做 graph 则 c ≈ 1.25–1.35**（以 decode 为 1）。

### 3.4 KV cache 与 SSM 状态回滚

| 状态 | 现状 | 回滚代价 |
|---|---|---|
| attention KV（`L.kc/L.vc`） | **位置索引**：`kv_write` 用 `pos = *pstart + j` 直接算 dst（`include/engine/kernels.h:350-363`）；attention 也是 `j <= qpos` 的因果实现（`:369-415`） | **免费**：回退只需把 `pos_dev_` 复位到已接受位置，多余槽位下次自动覆盖（`set_position()`，`src/runner.cpp:1424`） |
| MTP 层自己的 KV | 同上（新建，位置索引） | **免费**：被拒绝的 draft 槽位下次覆盖 |
| **SSM/线性注意力状态**（`conv_state`、`rec_state`） | **原地顺序推进**：`run_q35_linear_attn` 里 `k_gdconv_/k_gdstate_` 滚动 conv 历史、`gd_delta_rule*` 内核 `for (t=0;t<T;++t)` 逐 token 原地更新 `rec_state`（`src/qwen35.cpp:299-474`，`include/engine/kernels.h:1193/1247/1369`） | **必须显式处理**：9B 有 24/32 层是 SSM，35B 有 30/40 层 |

**推荐方案 A（P0，可预测）**：verify 前把每层 `conv_state`+`rec_state` 快照到 VRAM scratch，**并让 delta/conv 内核对 batch 的每个 token 步各写一份**（槽位 1..n_max；第 n_max+1 份就是 live 状态，不用额外拷）。接受 m 个 draft 后，把槽位 m+1（即"处理完 1+m 个 token 后"的状态）拷回。

- 单层状态量：`conv_state` = (K-1)×conv_dim×4B = 3×8192×4 = **98 KB**；`rec_state` = 32×128×128×4 = **2 MB**；合计 **≈2.1 MB/层**
- 9B：24 层 × 2 槽位（n_max=2）= **≈100 MB**，写流量 ≈100 MB/verify ≈ 0.4 ms
- 35B：30 层 × 2 槽位 = **≈126 MB** —— 但 35B 只剩 1137 MiB 且低于 margin，必须先降到 `gpu_moe_layers=14`

**方案 B（省显存，仅备选）**：只存一份 pre-verify 快照，部分接受时"恢复 + 重放已接受的 m+1 个 token"。
- 代价：α=0.6/k=2 时 P(部分接受)=0.64，平均要多跑 ≈0.8 次验证批前向 → **直接吃掉全部收益，不推荐**。

**方案 C（P2 优化，未验证）**：delta rule 更新是线性的（`S ← diag(exp(g))·S + β·k·vᵀ`），理论可逆：`S ← (S − β·k·vᵀ)/exp(g)`。逐元素即可，**零额外显存**；但 `exp(g)` 很小（g = −exp(A)·softplus(...)）时除法的数值放大需要实测，conv_state 另有 3 token 历史窗口可以重建。建议作为 P2 的可选加速，不要在 P0 引入。

### 3.5 收益模型（公式 + 阈值）

记号（全部以"一次普通 decode 前向"为 1 计价）：

- `k` = `--spec-draft-n-max`（每轮 draft 步数）
- `α` = 接受率（每步以 α 独立近似）
- `c` = 一次 k+1 token 验证前向的成本（单位 = decode）
- `d` = 一次 draft 步的成本（单位 = decode）

```
每轮产出 token 数  E = 1 + α + α² + ... + α^k          （k 个 draft + 1 个 bonus）
每轮耗时          T = c + k·d
加速比            S = E / T
不亏的条件        E > c + k·d
```

**阈值（本机估算，c/d 见 3.2/3.3）**：

| 场景 | c | d | k | α | S | 结论 |
|---|---|---|---|---|---|---|
| 9B 全 GPU，理想（有 verify graph） | 1.15 | 0.13 | 2 | 0.60 | **1.40** | ✅ 值得 |
| 9B 全 GPU，理想 | 1.15 | 0.13 | 2 | 0.73 | **1.61** | ✅ 值得 |
| 9B 全 GPU，**无** verify graph | 1.30 | 0.13 | 2 | 0.60 | **1.27** | ✅ 值得（P0 就能收） |
| 9B，k=1 | 1.15 | 0.13 | 1 | 0.60 | 1.25 | 次优（k=2 更好） |
| 9B，k=3 | 1.15 | 0.13 | 3 | 0.60 | 1.41 | 略优于 k=2，但 α 估计误差放大 |
| **35B-A3B（CPU 专家）** | **1.8**（见下） | 0.06 | 2 | 0.60 | **1.02** | ⚠️ 基本打平 |
| 35B-A3B | 1.8 | 0.06 | 2 | 0.45 | 0.86 | ❌ 负收益 |
| 现状（未做多行 kernel） | 13–15 | 0.13 | 2 | 0.73 | 0.15 | ❌ 完全不可行 |

**35B 的 c≈1.8 不是拍脑袋**：用用户自己的 llama.cpp 实测反解——`α=0.6157` 时 `E=1+α+α²=1.995`，实测只 `+6.6%` ⇒ `c+2d = 1.995/1.066 = 1.87`；取 `d=0.06` 得 **c ≈ 1.75**。机理也对得上：本引擎 `[moe-prof]` 实测 GPU 专家 kernel 的批成本——n=1 时 0.20 ms/层，**n=4 时 1.20 ms/层（单次 6×、折到每 token 1.5×）**；而 27 层 CPU 专家的 3 token 专家并集 ≈20/256，DDR 流量 ≈2.4×。**这个模型在 α=0.6 下只能给出 +2% 左右**。

**α 的经验范围（用户实测）**：temp 0.6/0.7 时 0.45–0.62，temp 1.0 时 0.39；greedy（本引擎 `run` 只有 argmax）应更高（**未验证**，估计 0.6–0.75）。**α<0.31（k=2, c=1.15）时必然亏**。

### 3.6 风险清单（逐条给应对）

| # | 风险 | 实测/读码依据 | 应对 |
|---|---|---|---|
| R1 | **n_tokens>1 路径不可用**（这是 1 号风险） | 9B n=3 = 326 ms vs n=1 = 22 ms；`matmul()` 整模型反量化 + f32 head | P0-A 多行 GEMV；**验收门槛：`T(3 token)/T(1 token) ≤ 1.25`** |
| R2 | MTP 层量化格式可能不被现有 GEMV 支持 | 35B：Q4_K/Q6_K/F16 router/Q8_0 head；Qwythos：Q8_0；均已在 `gemv` 支持列表（Q4_K/Q5_K/Q6_K/Q5_0/Q5_1/Q8_0），`Q5_1` 是唯一例外（本模型没有） | 多行 kernel 只做 Q4_K/Q5_K/Q6_K/Q8_0 四种；其余格式 fallback 到 1 行逐次（慢但对） |
| R3 | **SSM 状态回滚**（hybrid 模型必踩） | `rec_state`/`conv_state` 原地推进，无位置索引 | 3.4 方案 A（槽位快照）；用"vLLM 式全接受路径"做对照测试（flush+重算）验证数值 |
| R4 | 验证批不走 CUDA Graph → WDDM 提交开销 | `src/runner.cpp:1610` 只对 n==1 捕获；FINAL-REPORT：提交占 decode 20–30% | P1：为 `(n_tokens, position bucket)` 捕获 verify graph。注意 attention kernel 的 `nwarp` 依赖 `ctx_len`（`src/qwen35.cpp:581-585`），要按位置分桶（引擎已有 `attn_graph[3]` 桶机制可抄） |
| R5 | 35B 显存已满 | `post-load used=11150 MiB free=1137 MiB`，margin=1200 | 35B 的 MTP 层（545 MB）必须 CPU；且 `gpu_moe_layers` 16→14 会掉 ~5% decode → **建议 35B 直接不做，或明确接受"可能负收益"** |
| R6 | 采样（temp>0）下的接受判据不同 | 引擎 `run` 只有 argmax（`src/main.cpp:449-468`） | P0 只支持 greedy（判据 = 逐行 argmax 相等，精确且最简单）；temp>0 需要 `min(1, p_tgt/p_draft)` + 残差重采样，需要完整 logits 行 + sampler，放 P1 |
| R7 | 结构细节可能反推错（拼接顺序/是否再乘 mscale/是否用 `output_norm` 兜底） | 张量形状只能反推拓扑，不能反推顺序 | **验收用"接受率区间"**：真实文本 α 应在 0.4–0.8。α≈0（拼错）/α≈0.95+（重复退化，用户日志 R1 那组就是 0.959 的假象）都判为 bug |
| R8 | Q4_K decode 路径本身异常慢 | Qwythos（Q4_K_M，5.47 GB）32.4 ms vs Qwen3.5-9B（Q5_K_M，6.13 GB）23.3 ms，反直觉 40% | **未验证原因**（怀疑 kernel 选择/`kernel_select` 对 Q4_K 的形状判据）；建议作为独立 0.5 天任务先看，因为它直接决定 MTP 最佳候选的基线 |
| R9 | 与正在改 `src/main.cpp` 的智能体冲突 | 当前有另一个 agent 在做 HTTP 服务 | spec 循环放到**新文件 `src/spec.cpp`**（或 `src/runner_spec.cpp`），`main.cpp` 只留 3 行接线；`build.sh` 的构建列表同步加一行（需与对方协调） |

### 3.7 接口草案（可直接落地）

```cpp
// include/engine/runner.h
struct SpecConfig {                 // 来自 config.cpp: spec_type/spec_draft_n_max/spec_draft_p_min
  bool enabled = false;
  int n_max = 2;
  float p_min = 0.0f;
};
struct SpecStats { int rounds=0, drafted=0, accepted=0; double mean_accept_len=0; };

class DenseRunner {
 public:
  // 一次性跑 R 轮 spec decode，返回生成的 token；greedy。
  std::vector<int> generate_spec(const std::vector<int>& prompt, int n_gen,
                                 const SpecConfig& cfg, SpecStats* stats);
  bool has_mtp() const { return mtp_.valid; }
 private:
  struct MtpW {
    bool valid = false;
    int block = -1;
    LayerW layer;                    // 复用 run_q35_attn_attn / run_ffn / run_moe
    QTensor eh_proj;
    cuda::Buffer enorm, hnorm, shared_head_norm;
    cuda::Buffer kc, vc;             // 1 层 KV，位置索引
    cuda::Buffer ssm_snap;           // 3.4 方案 A 的槽位快照（整模型一份）
    std::vector<cuda::Buffer*> ssm_conv, ssm_rec;  // 便于快照/恢复
  } mtp_;
  // 多行验证：返回每行 argmax（logit_rows = tokens.size()）
  std::vector<int> forward_verify(const std::vector<int>& tokens, int start_pos);
};
```

主循环（greedy，k 步链式 draft）：

```
pos = 已提交位置; t_next = argmax(logits@pos)      # 已有的 bonus token
d[0..k-1] = draft_chain(h_p, t_next, pos)          # MTP n=1 前向，k 步
batch = [t_next, d[0], ..., d[k-1]]                # 1+k 个 token，位置 pos

snapshot_ssm()                                     # 3.4 方案 A（写槽位 1..k）
A = forward_verify(batch, pos)                     # 多行 GEMV，拿 1+k 个 argmax
m = 最长匹配前缀（A[i] == batch[i+1]）
accept = batch[0..m]                               # 至少 1 个
t_next = A[m]                                      # 新的 bonus
restore_ssm(m+1)                                   # 恢复"处理完 1+m 个 token"的状态
pos += m + 1
```

---

## 4. 工作量与优先级

### P0 —— 最小可用（单步/双步 draft + 验证）
| 子项 | 文件 | 风险 |
|---|---|---|
| P0-A 多行量化 GEMV + 多行 head + `forward_verify` | `include/engine/kernels.h`、`src/runner.cpp`、`include/engine/runner.h`、`src/kernel_select.cpp` | **中**（纯加法，有 `build/fixtures/gemv_*` 现成对拍夹具；数值必须与 1 行 kernel 逐位一致） |
| P0-B IR/装载 MTP 层 | `src/ir.cpp`、`include/engine/ir.h`、`src/runner.cpp` | 低（照 `load_q35_attn` 抄） |
| P0-C draft 路径（eh_proj + MTP 层 + 共享 head） | `src/spec.cpp`（新）、`src/runner.cpp` | 中（拼接顺序未验证） |
| P0-D 验证+接受+SSM 回滚 | `src/spec.cpp`、`src/qwen35.cpp`、`include/engine/kernels.h` | **中高**（SSM 槽位快照要改 2–3 个 kernel 的签名） |
| P0-E 配置/CLI/统计 | `src/config.cpp`、`docs/CONFIG.md`、`src/main.cpp`(3 行) | 低（**注意 main.cpp 正被别人改**） |
| P0-F 验收 | 新测试 + 真实文本 A/B | 低 |

**P0 验收线**：① `T(3 token verify)/T(1 token decode) ≤ 1.25`；② 真实中文文本 α ∈ [0.4, 0.8]；③ 生成结果与 `n_max=0`（纯 greedy）**逐 token 一致**（投机解码必须是无损的，除非实现采样）。

### P1 —— 链式 k 步 + 自适应 + CUDA Graph
- k 从 1 → 2/3；**用滑动 α 均值自适应 k**（`α>0.7` 且 `k<3` 升；`α<0.35` 降），避免 n-max=4 那种负优化（用户实测 20.36 vs 25.76）。
- verify CUDA Graph：按 `(n_tokens ∈ {2,3,4}, position bucket)` 捕获，复用 `attn_graph[3]` 的桶思路。
- 温度/采样支持（需要完整 logits 行 + 标准接受判据 + 残差重采样）。
- 文件：`src/spec.cpp`、`src/qwen35.cpp`（graph 捕获）、`src/runner.cpp`。

### P2 —— 与 CUDA Graph/并发配合、可选捷径
- 方案 C（可逆 delta 回滚）替代 100 MB 快照。
- **ngram / prompt-lookup 自草稿**：不需要 MTP 头、四个模型都能用、复用的正是 P0 的 verify 机制；对重复性文本（agent 工具调用、代码编辑回显）收益可能很可观。llama.cpp 里有一整族（`--spec-type ngram-*`），值得作为"零权重成本"的并行赌注。
- 多序列/并发下的 spec（用户实测 MTP 与并发是**替代关系**不是叠加：4 并发时 120.77 vs 122.58 → **单流才开 MTP**）。

### 明确回答："最可能在哪个模型上拿到多少提升？值不值得先做？"

| 候选 | 引擎现状 | MTP 后预估 | 依据 |
|---|---|---|---|
| **Qwythos-9B-MTP**（额外发现，引擎已能加载） | 32.4 ms/token = **31 tok/s** | **41–48 tok/s（+34% ~ +55%）** | 全 GPU 无 CPU 专家（c≈1.15–1.3）；llama.cpp 同模型实测 α=0.67–0.73、+25%~+43% |
| Qwen3.6-35B-A3B（四个模型里唯一有 MTP） | 38.9 ms = **25.7 tok/s**（用户口径） | **26–27.5 tok/s（+0~7%）** | c≈1.8（GPU 专家 n=4 时单次 6×、27 层 CPU 专家且 3 token 专家并集翻倍）；显存只剩 1137 MiB，MTP 层 545 MB 必须 CPU；llama.cpp 同模型实测只有 +6.6% |
| Qwen3.5-9B / Coder-V2-Lite / qwen2.5-0.5b | 46.4 / 23.4 / 197 tok/s | 无 MTP 头 → 本轮 0 | 1.2 节数据 |

**结论：值得做，但目标不是那四个模型里唯一带 MTP 的 35B，而是 Qwythos-9B-MTP。** 原因：
1. 它是唯一"有 MTP 头 + 全 GPU 驻留 + 高接受率"的组合；
2. 引擎**今天就能加载它**（实测 VALIDATION PASS、31 tok/s），MTP 只是"再加一层 + 一个验证批"；
3. 稠密全 GPU 模型的 MTP 收益在用户自己的实测里是 +25%~+43%，而 MoE+CPU 专家的 35B 只有 +6.6%。

**但先做什么？** 按 3.3 的验收门槛：**先做 P0-A（多行量化 GEMV + 多行 head），它同时是 MTP 的唯一硬门槛和小 batch prefill 的加速器**。做完测一个数：

- `T(3 token verify)/T(1 token decode) ≤ 1.25` → 立刻上 Qwythos-9B 的 P0-B/C/D，预期 +34%~+55%；
- `≥ 1.5` → **MTP 在这台机器上不值得继续投入**（35B 更不可能），把 P0-A 收作 prefill 优化即可，收益仍然真实（短 prompt 那 0.33 s 固定成本会消失）。

---

## 5. 未能确认的事实（"未验证"清单）

1. **`eh_proj` 的拼接顺序**（`[enorm(emb) ; hnorm(h)]` 还是反过来）、以及 `shared_head_norm` 是否在 head 前、是否还要乘 rope/mscale —— 张量形状只能反推拓扑，**顺序未验证**。只能靠"接受率是否落在 0.4–0.8"和与 llama.cpp 输出对拍来定。
2. **llama.cpp `--spec-draft-n-max` 的默认值**：`local-llm-engine-p0` 记忆记为 b12128 默认 3；用户日志只说 b11718 起"默认 n-max 变大"。字符串里只读到 flag 名与 `n_max/n_min` 日志格式，**具体默认值未验证**。
3. **llama.cpp recurrent partial rollback 的具体实现**（`n_rs_seq` 快照槽位？按 cell 存状态？）：只读到 `llama_memory_recurrent`、`rollback=%u`、`DSV4 recurrent state rollback index` 等字符串与错误断言，**机制细节未验证**；本报告的 3.4 方案是我方设计，不是对 llama.cpp 的复刻。
4. **llama.cpp MTP 是否在 verify 批里把已接受 token 回填到草稿 KV、以及草稿 KV 的确切位置语义**：只知草稿侧是独立 context（自己的 KV/放置参数），**位置语义未验证**。
5. **本机实测的绝对 ms 偏低**：GPU 空闲时钟（nvidia-smi 见到 210–435 MHz）且没有 `--kernel-calibrate` 缓存，9B decode 测得 22.2–23.1 ms（43–45 tok/s），比用户口径 21.6 ms / 46.4 tok/s 慢 5–8%；35B 测得 57.8 ms/token，比用户口径 38.9 ms 慢 48%。**报告的比值（batch/decode、d/decode）是主证据，绝对值仅供量级参考**。
6. **n>1 固定成本 330–357 ms 的拆分**（整模型反量化 vs f32 lm_head 物化各占多少）：每次测量都是新进程、且只跑一次 n>1 前向，两者混在一起，**未拆分**。
7. **Qwythos（Q4_K_M）比 Qwen3.5-9B（Q5_K_M）慢 40% 的原因**：反直觉（字节更少反而更慢），怀疑 Q4_K kernel 选择或 yarn rope 路径，**未验证**。
8. **MTP 层的 MoE 在 35B 上的实际 CPU 代价**：由权重字节数和 DDR 带宽估算（≈1.3 ms/步），**未实测**。
9. **greedy（argmax）下的真实接受率**：用户所有实测都在 temp 0.6–1.0；引擎只有 greedy，预期 α 更高但**未实测**。
10. **`--quant-gemm` 为什么慢**（568 ms）：只测到结果，未 profile 根因。

---

## 6. 复现命令（只读，不写仓库）

```bash
# 1) 四个模型 + Qwythos 的 MTP 头盘点（仓库外脚本，只读 GGUF 头）
python <仓库外>/gguf_probe.py <model.gguf> meta.json   # 本次侦察用的只读 GGUF 探针（约 120 行：读 header+KV+tensor info，不读张量数据）
# 2) 引擎侧"检测到但丢弃"的证据
./build/engine.exe info "<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
#    -> roles: ... other:20 ; warn: MTP head detected on blocks; excluded from 40 layers
# 3) 单步 vs 批前向的成本（本报告最关键的一组数）
ENGINE_DBG_FWD=1 ./build/engine.exe run --model <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf --ids 760 --n 8 --ctx 512
ENGINE_DBG_FWD=1 ./build/engine.exe run --model <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf --ids 760,761,762 --n 1 --ctx 512
# 4) 35B 的 MoE 批放大（n=1 0.20 ms/层 → n=4 1.20 ms/层）
ENGINE_PROFILE=1 ./build/engine.exe run --model <35B.gguf> --ids 760 --n 8 --ctx 512 2>&1 | grep moe-prof
# 5) Qwythos-9B-MTP 在引擎上"能加载、能跑、只是不用 MTP"
./build/engine.exe run --model <MODEL_DIR>/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf --ids 760 --n 8 --ctx 512
```

数据文件：`<MODEL_DIR>\AI-最优启动\实测原始数据.txt`（184 组压测，含全部 MTP 接受率/超参）、`<MODEL_DIR>\AI-最优启动\3-Qwythos-9B-MTP-192K.bat`、`13/14/21-*Huihui-35B*.bat`（`--spec-type draft-mtp --spec-draft-n-max 2`）。
