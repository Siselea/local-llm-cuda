## 改了什么 / What changed

<!-- 一句话概述；关联 issue: Fixes #... -->

## 类型 / Type

- [ ] Bug fix
- [ ] Performance
- [ ] New feature
- [ ] Docs / CI / tooling only
- [ ] Refactor (无行为变化)

## 验证 / Verification

- [ ] `bash tools/verify.sh` → ALL GREEN
- [ ] `bash tools/smoke_models.sh` → 四模型通过
- [ ] `bash tools/gen_regression.sh` → 10/10
- [ ] `bash tools/verify_generality.sh` → GREEN
- [ ] 仅文档/CI 改动（可不跑上述四套，请在下方说明）

命令与结果摘要（必填）：

```text
# 粘贴关键 PASS/FAIL 行
```

## 性能/显存影响 / Performance impact

<!-- 性能类 PR 必填：前后对比表、复现命令、GPU 状态（空闲/占用）、ENGINE_KPROF=1 数据 -->

## 兼容性 / Compatibility

- [ ] 不影响现有配置项/CLI
- [ ] 新增/修改配置项已同步 `src/config.cpp` schema 与 `docs/CONFIG.md`
- [ ] 不改推理/内核行为（或已给出对拍数据）

## 检查 / Checklist

- [ ] 没有提交 `build/`、二进制、模型权重、日志、本机绝对路径
- [ ] 没有按模型名硬编码特判
- [ ] 新 CUDA kernel 走 NVRTC（`include/engine/kernels.h`）
- [ ] CI 通过（`build` / `sanity` / `no-private-paths`）
