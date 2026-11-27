# 外部审查 v0.1.7 · 缺陷清单核对（模型加载与数值正确性组）

> 核对人：prefill-fast｜日期：2026-11-27｜范围：审查清单 #8/#9/#10/#13/#14/#15/#16/#17/#18/#19/#26/#27
> 规则：先核实真假 → 真的修（带测试）→ 假的给证据。**结论：11 条 REAL（其中 2 条高危崩溃 #8/#10、1 条高危数据损坏 #16），1 条 REAL-but-MITIGATED（#27）；没有一条是误报，但有一条的"已修"前提是错的（#19）。**
> 代码：`src/gguf.cpp`、`src/st.cpp`、`src/ir.cpp`、`src/planner.cpp`、`src/runner.cpp` 的加载/放置函数 + 新的 `include/engine/fp16.h`（host/device 共用的 fp16 转换）。
> 测试：`tests/test_fp16.cpp`、`tests/test_loader_bounds.cpp`、`tests/test_gguf_moe_offload.cpp`（全部纳入 `build.sh` 与 `tools/verify.sh` 门禁）；合成坏文件，不需要大模型、不占 GPU。

---

## 0. 汇总

| # | 判定 | 缺陷 | 修复 | 测试 |
|---|---|---|---|---|
| 8 | **REAL（高危崩溃）** | `general.alignment` 转 `uint32` 截断为 0 → 除零（SIGFPE） | gguf.cpp 校验对齐值（2 的幂、≤64 MiB）后才转换；溢出安全 padding；data_offset/张量范围校验 | `test_loader_bounds` alignment 用例（预修复版实测 `0xC0000094`） |
| 9 | **REAL（数据损坏）** | 张量维度乘积 / nbytes 无溢出检查 | 逐步 checked multiply；nbytes 溢出检查；张量 offset+nbytes vs 文件大小 | dims_overflow / nbytes_overflow / offset_oob / offset_wrap |
| 10 | **REAL（高危崩溃，报告点名）** | GGUF F16/F32 MoE 回退不填 CPU 专家缓冲、无视放置 | 回退路径按 gpu_n 放置：CPU 专家保留 host 副本、只上传 GPU 专家、设置 cpu_begin/h_type_* | `test_gguf_moe_offload`（F16+F32 模型，GPU/全 CPU/混合三种放置；预修复版 all-CPU 实测 `0xC0000005`） |
| 13 | **REAL（几何未校验）** | GQA `n_head % n_head_kv`、GDN `inner % tsr` 无整除校验 | ir.cpp 加载期抛错：head 整除、GDN 可整除且 tsr>0、量级护栏 | `test_loader_bounds` 覆盖 GGUF 层；ir 用例随现有 IR 测试（见 §13） |
| 14 | **REAL（接受非法元数据）** | safetensors 负数/小数 shape、offset 被 `static_cast<uint64_t>` | 严格 `as_index()`：必须是有限非负整数、≤2^53 | `test_loader_bounds` st_neg_dim / st_frac_dim / st_neg_off / st_frac_off |
| 15 | **REAL（数据损坏）** | safetensors size/offset 算术无溢出检查 | `checked_mul()`、`end<begin` 拒绝、溢出安全的文件边界检查 | st_overflow / st_end_lt_begin / st_oob |
| 16 | **REAL（数值错误，高危数据损坏）** | host F16 解码把**次正规数**当正规数（最小次正规差 **512×**） | 新建 `include/engine/fp16.h`（host 实现 + 同一份 device 源码）；`kernels::kQuantCommon` 改为组合该源码；两处 host 解码改调共用函数 | `test_fp16`：**65536 个 pattern 全部逐位比对**（host vs 独立算术参考 + NVRTC 编译的 device 实现）、往返恒等、边界值；131090 checks |
| 17 | **REAL（静默 I/O 失败）** | 张量读取不检查 `gcount()`，截断文件被静默接受 | 新增 `read_exact()` 覆盖 runner.cpp 6 处读取；st.cpp `read_text` 同样检查 | 由坏文件/截断路径覆盖（代码级） |
| 18 | **REAL（DoS/巨大分配）** | `n_tensors`/`n_kv` 无上限就 reserve/循环；数组 payload 不与文件大小核对 | 计数按文件大小上界校验；数组 payload 超过剩余文件大小即拒 | `test_loader_bounds` 计数用例 |
| 19 | **REAL（并非"已在 v0.1.6 修过"）** | GGUF 嵌套元数据数组无递归深度上限 | `read_array/read_value` 增加 depth 参数，上限 16 | deep-array 用例（深 8 接受、深 4096 拒绝） |
| 26 | **REAL（显存浪费）** | safetensors MoE 即使保留 CPU 副本也全部 upload | 只上传 GPU 常驻前缀，CPU 剩余保留 host 副本并设 cpu_begin | `test_gguf_moe_offload` 的混合放置契约（gpu=1/host=4） |
| 27 | **REAL-but-MITIGATED** | planner 用"平均专家层大小"估 GPU 层数 | 运行时已被**实测逐层校验**（commit `8199927`、`388106a`）纠正；planner 数字仅用于展示（`engine plan`/launcher） | 见 §27（不建议本轮改：需同步 `pyref/planner.py` 的逐字节 parity） |

---

## #8 对齐值截断 → 除零（REAL，高危崩溃）

**位置**：`src/gguf.cpp:293-296`（v0.1.7 HEAD：`int64_t align = m.get_int("general.alignment", 32); m.alignment = static_cast<uint32_t>(align > 0 ? align : 32); ... (pos + m.alignment - 1) / m.alignment ...`）。
**机理**：`general.alignment` 是不可信元数据且读取为 `int64`；`align > 0` 只挡住负数/零，`2^32` 通过检查后被 `static_cast<uint32_t>` 截断成 **0**，随后 `(pos + 0 - 1) / 0` 触发整数除零。
**证据**：合成 GGUF（`general.alignment = 2^32`）喂给修复前二进制 ⇒ `test_loader_bounds.exe` 退出码 **`0xC0000094`（INTEGER_DIVIDE_BY_ZERO）**；修复后同一输入被干净拒绝。
**修复**：先校验 `1 ≤ align ≤ 64 MiB` 且为 2 的幂（非法则抛错）；padding 用溢出安全写法；`data_offset ≤ file_size`；每个张量 `offset+nbytes` 必须落在文件内。另删掉原来"静默把非法对齐改回 32"的行为（那会把坏文件当合法读）。

## #9 张量维度乘法溢出（REAL，数据损坏）

**位置**：`src/gguf.cpp:285-289`（`t.n_elements *= d;`、`t.nbytes = t.n_elements / block * bytes`）。
**机理**：维度来自文件；`[2^32, 2^32]`（uint64）乘积回绕成 0 ⇒ `nbytes = 0` ⇒ 张量表"合法"但实际数据不存在，后续读取越界/静默错值；nbytes 的 `/block*bytes` 同样可回绕。
**修复**：`t.n_elements *= d` 改为 checked multiply（`d == 0` 拒绝、`n_elements > MAX/d` 拒绝）；`n_blocks > MAX/bytes` 拒绝；解析完 `data_offset` 后逐张量校验 `offset + nbytes ≤ file_size - data_offset`（溢出安全形式）。
**测试**：`test_loader_bounds` 的 dims_overflow / nbytes_overflow / bad_block / offset_oob / offset_wrap 全部被拒。

## #10 GGUF F16/F32 MoE 回退不填 CPU 专家缓冲（REAL，高危崩溃，**报告点名项**）

**触发条件（精确）**：MoE GGUF 且其专家张量类型走 `runner.cpp` 的 **F32 fallback 分支**——即 `quant_kernel(type_name) == nullptr`。实测 **F16 与 F32 专家组都会进入该分支**（`ENGINE_DBG_LOAD2=1`: `[load2] layer 0 -> F32 fallback (gpu_n=4/4)`；F16 也打印同样内容），所以"F16/F32 GGUF 的 MoE 模型 + 单卡放不下全部专家"即触发。
**位置**：`src/runner.cpp:1949-1966`（v0.1.7 HEAD）：
```cpp
} else {                                   // F16/F32 专家组
  ... w.experts.resize(n_expert);
  for (int e = 0; e < n_expert; ++e) { w.experts[e].gate = upload(g); ... }   // 全部上传
}
```
**两个后果**：
1. 不填 `w.host_gate/host_up/host_down`（GGUF 分支没有任何地方填 f32 host 副本）且不设 `w.cpu_begin`/`h_type_*`；
2. 无视分层放置（`gpu_n`/`gpu_moe_layers_`）把全部专家上传 GPU ⇒ 大模型直接 cuMemAlloc 失败/超订，而 `run_moe` 的 CPU 分支在需要 CPU 专家时按 `L.host_gate[e]` 索引**空 vector** ⇒ 越界/崩溃。
**证据（预修复版实测）**：F32 专家 tiny MoE（4 专家）× `--gpu-layers 0`（专家全放 CPU）⇒ 退出码 **`0xC0000005`（ACCESS_VIOLATION）**；修复后同配置通过（rel_l2≈2.6e-7，与 numpy 参考一致）。
**修复**：回退分支与量化路径对齐——按 `gpu_n` resize `w.experts`、只上传 `e < gpu_n` 的专家、`gpu_n < ne` 时保留全部 f32 host 副本、设置 `w.cpu_begin = gpu_n` 和 `h_type_*`（CPU 行内核需要类型名）。
**测试**：`tests/test_gguf_moe_offload.cpp` 用 `tests/make_tiny_gguf_moe.py` 的 `EXPERT_TYPE=F16|F32` 生成模型 + numpy 参考 logits，跑 **GPU / 全 CPU / 混合(gpu_experts=1)** 三种放置，既比数值（rel_l2）又断言放置契约（`debug_expert_placement`：gpu/host 数、cpu_begin）。

## #14 safetensors 负数/小数维度与偏移（REAL，接受非法元数据）

**位置**：`src/st.cpp:81-85`：`shape.push_back(static_cast<uint64_t>(d.num))`、`begin/end = static_cast<uint64_t>(offs.arr[i].num)`。
**机理**：safetensors 头是 JSON，数字经 `strtod` 成 `double`。`-1` → `0xFFFF...`（巨大），`1.5` → 静默截断，`>2^53` 丢精度，`NaN/inf` 的 double→uint64 转换是 UB。原代码无任何检查。
**修复**：新增 `as_index()`：必须是 number、有限、`≥0`、整数、`≤2^53`，否则抛错；shape 与 data_offsets 四处全部走它。
**测试**：st_neg_dim / st_frac_dim / st_neg_off / st_frac_off 被拒；合法头仍接受。

## #15 safetensors 大小/偏移算术溢出（REAL，数据损坏）

**位置**：`src/st.cpp:87-91`：`expect = n_elements() * dtype_bytes`、`t.end - t.begin`、`data_start + t.end > fsize`。
**机理**：shape 乘积与 `× dtype_bytes` 均可回绕；`end<begin` 时 `end-begin` 回绕；`data_start+end` 可回绕后绕过文件大小检查 ⇒ 接受不可能的张量 → 后续 seek/read 越界。
**修复**：`checked_mul()`；`end < begin` 直接拒绝；用 `t.end > fsize || data_start > fsize || data_start + t.end > fsize`（先比较再相加）防回绕。
**测试**：st_overflow / st_end_lt_begin / st_oob。

## #16 F16 次正规数转换错误（REAL，高危数据损坏；lead 已定位，逐位验证见下）

**位置（两处，均 host 解析路径）**：`src/runner.cpp:1364`（safetensors F16）与 `src/runner.cpp:1405`（GGUF F16）。
**错误**：`if (exp == 0) bits = sign | (mant ? (112u << 23) | (mant << 13) : 0);` —— 把 fp16 次正规数当成正规数处理，返回 `2^-15 * (1 + mant/1024)`；正确值是 `mant * 2^-24`。最小次正规（mant=1）：**3.05e-5 vs 5.96e-8，差 512×**。影响所有走 host 解码的 F16 权重（F16 GGUF 的 CPU 专家、F16 embedding 等）的小值区间。
**修复**：
- 新建 `include/engine/fp16.h`：**唯一的转换定义文件**——host `half_to_float_bits()`/`half_to_float()`/`float_to_half_bits()`（含次正规重规格化、NaN payload 保留）+ `fp16::kDeviceSrc`（NVRTC 编译用的同一份 device 实现，紧挨着 host 实现）；
- `include/engine/kernels.h` 的 `kQuantCommon` 改为 `engine::fp16::kDeviceSrc + R"cuda(...)"`（不再自带一份 `half_to_float`）⇒ 设备端与 host 端来自**同一处文本**；
- 两处 host 解码改调 `engine::fp16::half_to_float()`，并与 BF16 分支拆开（BF16 仍是 `<<16`）。
**逐位验证方法（`tests/test_fp16.cpp`，131090 checks PASS）**：
1. **全部 65536 个 fp16 bit pattern** 与"独立算术参考"比对：参考用 `ldexp` 按数值定义计算（不是位操作复制，避免同错），有限值要求 **bit-for-bit 相同**，NaN 要求 class+符号一致、payload 非零；实测 fp16 有 **2046 个 NaN pattern**（1023 mantissa × 2 符号）；
2. 全部非 NaN pattern **half→float→half 恒等**（half→float 精确，float→half 最近偶舍入应还原）；
3. 边界用例：`0x0001`（2^-24，bits `0x33800000`）、`0x03FF`（1023×2^-24）、`0x8001`、±0、`0x3C00`=1.0、`0x7BFF`=65504、±inf、`0x0400`（2^-14 最小正规数）；
4. **设备/主端一致性**：用 `cuda::compile_ptx(fp16::kDeviceSrc + probe kernel)` 编译 NVRTC 版本，对 65536 个 pattern 与 host 实现逐位比对（GPU 可用时执行，CI 无 GPU 时跳过）；
5. 接线检查：`kernels::kQuantCommon` 中 `half_to_float` 恰好定义 1 次且包含次正规重规格化循环（防止两份实现再次分叉）。
（`cpu-gemv-bw` 的 F16C 工作针对 CPU 块头解码，与本次 host/载入路径无关；本测试的参考法可直接复用其思路。）

## #17 张量读取缺完整性检查（REAL，静默 I/O 失败）

**位置**：`src/runner.cpp` 6 处 `f.read(...)`（safetensors F32/F16 张量、GGUF tensor_f32/load_qtensor/read_gguf_raw 两处）均未检查 `gcount()`；`ifstream::read` 在文件提前结束时**不清空目标缓冲**，于是截断/被并发修改的模型文件会"成功"返回零或旧字节。
**修复**：新增 `read_exact(std::ifstream&, void*, size_t, what)`，`gcount() != bytes` 即抛错；6 处全部替换。`src/st.cpp` 的 `read_text()` 同样加短读检查。**注**：`gguf::Reader::read_raw()` 原本就检查 gcount（审查者把 #17 笼统写成 gguf，实际风险在 runner 的张量读取）。

## #18 不可信计数触发巨大分配（REAL，DoS）

**位置**：`src/gguf.cpp:255-269`：`m.n_tensors`/`m.n_kv` 直接来自文件，`m.tensors.reserve(static_cast<size_t>(n_tensors))` 先于任何校验；`n_kv=2^60` 之类会让 `reserve` 申请多 EB（bad_alloc/length_error 崩溃）；`read_array` 的 `v.raw.resize(array_len * elem)` 只受常量 1e8 限制（800 MB）且与文件实际剩余字节无关。
**修复**：按"文件大小 / 每条记录最小字节"给计数加上界（张量记录 ≥24 B、kv 记录 ≥16 B），超标即抛错并说明；`read_array` 的 payload 先与 `r.size()` 比对，超过文件大小即拒。
**测试**：`n_tensors = {2^20, 2^40, ~0}` 的合成头全部被干净拒绝。

## #19 嵌套元数据数组无递归深度限制（REAL —— **"已在 v0.1.6 修过"的前提不成立**）

**核实**：v0.1.6 的深度上限修复是 **JSON 解析器**（commit `e0f79d3 fix(serve): cap JSON nesting depth`，针对 `POST /tokenize` 深层 `[` 打崩进程）；GGUF 元数据读取器**没有**对应修复——`git show HEAD:src/gguf.cpp` 中 `read_array(Reader&, bool)`（第 71 行）无 depth 参数，第 121 行 `read_array(r, false)` 直接无界递归。攻击面：恶意 GGUF 的嵌套 ARRAY 元数据可递归到栈溢出。
**修复**：`read_array`/`read_value` 增加 depth 参数，`kMaxArrayDepth = 16`，超限抛错（与 JSON 侧同一策略）。
**测试**：深 8 的数组被接受（不误杀真实模型），深 4096 被拒。

## #26 safetensors MoE 专家重复驻留（REAL，显存浪费）

**位置**：`src/runner.cpp` safetensors MoE 分支（`w.router = ...` 之后）：`if (gpu_experts_ >= 0) { w.host_gate.push_back(g); ... }`（保留 CPU 副本）之后**无条件** `w.experts[e].gate = upload(g)` ⇒ 为 CPU 保留的专家同时驻留 host 与 device。
**修复**：按 `gpu_n = (gpu_experts_ < 0) ? ne : min(gpu_experts_, ne)` 只上传前 `gpu_n` 个专家；`gpu_n < ne` 时保留全部 host 副本（CPU 分支按 expert id 索引，必须全量）并设 `cpu_begin`/`h_type_*`。自动放置（`gpu_experts_ = -1`）行为不变（全部 GPU）。
**验证**：`test_gguf_moe_offload` 的放置契约（混合模式 gpu=1 / host=4 / cpu_begin=1）覆盖同一套加载契约；safetensors 路径复用相同断言逻辑（同一 hook）。

## #27 planner 平均专家层大小（REAL，但运行时已缓解）

**位置**：`src/planner.cpp:309-350`：`gpu_moe_layers = vram_left / avg_layer`（avg = expert_bytes / n_moe_layers）。各层专家字节可能不同（dense 前置层、shared expert 等），平均会估错"能在 GPU 放几层"。
**缓解（当前机制已覆盖实际放置）**：
- `src/runner.cpp` 的自动分层在装载**每一层前**用**实测 free VRAM** 校验（`expert_layer_fits_vram`，commit `8199927` 加入 `vram_extra_reserve_mb` 标定预留 + `388106a` 加入"按实测显存收敛/只允许连续前缀扩展"），不是按平均值决定；
- 装载后的 `[vram] check OK`（margin）与 ledger-gap 输出进一步兜底。
- planner 的数字只用于 `engine plan`/launcher 展示（`tools/diff_plan.py` 对比 `pyref/planner.py` 的逐字节 parity 是本仓库门禁的一部分）。
**结论**：**REAL（展示/估算不精确）但运行时已被实测校验纠正**；彻底修需要同时改 `planner.cpp` 与 `pyref/planner.py` 的算法（保持 parity 门禁），属后续任务，本轮不改。

## #13 Gated DeltaNet / GQA 几何整除性校验（ir 部分 REAL）

**位置**：`src/ir.cpp:271-289`（GDN 几何）与 `:157-179`（GQA head 几何）。
**缺失的校验**（全部为"坏/不支持元数据静默接受"）：
1. `n_head % n_head_kv` 未校验——GQA 头映射 `h / (n_head/n_kv)` 整数除法会静默丢头、读错 K/V 行；
2. `value_head_dim = inner / tsr` 用整数除法，`inner % tsr != 0` 静默截断；`tsr <= 0` 产生 0 头维；
3. `hidden % n_head` 未校验（`key_length` 回退计算会截断）；
4. GDN 量级无护栏（conv_kernel/groups/state 异常大）。
**修复**：`ir.cpp` 加载期抛 `std::runtime_error`（GQA head 计数关系、GDN `inner % tsr == 0` 与 `tsr > 0`、`hidden % n_head`、非法量级），错误信息指出具体字段。
**归属**：`qwen35.cpp` 的**运行时**卷积/混合头几何（per-layer 不同 head 数、conv 状态索引）属 `edge-kernels` 执行路径，本轮只做了 **IR 加载期校验**；已同步给 lead 转交。

---

## 验证

| 项 | 结果 |
|---|---|
| 干净 worktree 构建（`build.sh`） | ✅ 通过（含新增 3 个测试目标） |
| `test_fp16.exe` | ✅ 131090 checks PASS（65536 pattern host vs 独立参考 + 往返 + 设备逐位一致） |
| `test_loader_bounds.exe` | ✅ **28 checks PASS**（GGUF/safetensors 恶意文件 + IR 几何 4 例全部干净拒绝） |
| `test_gguf_moe_offload.exe`（F16 / F32 模型） | ✅ PASS（GPU / 全 CPU / 混合放置，数值 rel_l2≈2.6e-7，放置契约成立） |
| `verify.sh`（五套之一，含新增门禁） | 见提交说明（本轮运行结果附后） |
| 修复前证据 | #8 `0xC0000094`（除零）、#10 `0xC0000005`（越界访问）均在合成输入上复现 |

## 未纳入本轮的观察（交 owner）

- `include/engine/kernels.h` 的 device `f32_to_f16_bits`（`kF32ToHalf`）对 **f32→f16 次正规结果做 flush-to-zero**（`if (e <= 0) return sign;`）。它作用于**激活值**（prefill 的 x/slab 转换），不是权重；影响是极小激活被截为 0，与 #16（解码）不同。该字符串在 prefill 热路径上，属 `edge-kernels`/`cpu-gemv-bw` 的修改范围，建议他们评估是否改用 `cvt.rn.f16.f32`（会轻微改变 prefill 数值，需要重新对拍）。
