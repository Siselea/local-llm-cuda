# 源码归档规范（source archive）

v0.1.7 的"完整源码归档"由手工拼装，结果**不可构建**（缺 `build.sh`/`README.md`/`LICENSE` 等根文件）且混入 50 个编辑器备份与 `__pycache__`。自 v0.1.8 起，源码归档一律由工具生成，规则如下。

## 1. 生成方式（唯一入口）

```bash
python tools/make_source_archive.py            # -> dist/local-llm-cuda-<tag>-src.zip
python tools/make_source_archive.py --list     # 预览将被打包的文件
python tools/make_source_archive.py --out D:/x.zip
```

- 归档根目录：`local-llm-cuda-<tag>/`（tag 取 `git describe --tags --abbrev=0`，未打 tag 时用 `HEAD`）
- 打包后工具会自检：若归档中仍出现 `.pre*` / `.v1` / `__pycache__` 会返回非 0

## 2. 必须包含（白名单）

| 类别 | 内容 |
|---|---|
| 根文件 | `build.sh`、`README.md`、`LICENSE`、`CONTRIBUTING.md`、`THIRD-PARTY.md`、`engine.cfg.example` |
| 源码树 | `src/`、`include/` |
| 测试 | `tests/` |
| 工具 | `tools/` |
| Python 参考实现 | `pyref/` |
| 文档 | `docs/`（不含构建产物） |
| CI | `.github/` |

缺任一必需根文件时，工具直接报错退出（`exit 2`），**不允许发布残缺源码包**。

## 3. 必须排除

| 排除项 | 理由 |
|---|---|
| `*.pre`、`*.pre-*`、`*.pre_*`、`*.v1`、`*.bak`、`*.orig`、`*.rej` | 编辑器/手工备份，非源码 |
| `__pycache__/`、`*.pyc`、`*.pyo` | 与解释器版本绑定的字节码 |
| `build/`、`dist/`、`.git/`、`.pytest_cache/`、`.mypy_cache/` | 构建产物与仓库元数据 |
| `nohup.out`、`*.log`（根目录临时日志） | 运行残留 |

## 4. 验收（发布前必做）

1. 在**空目录**解包生成的 zip；
2. 按 README 的快速开始实跑一次 `bash build.sh`；
3. 确认产物可运行（`engine.exe --help`）；
4. 把 zip 大小与 SHA256 一并写进 Release notes。

> 依据：v0.1.7 归档缺 `build.sh` 导致"完整源码"不可构建，属发布级缺陷（见
> [DEFECT-SUMMARY-v0.1.7.md](reports/DEFECT-SUMMARY-v0.1.7.md) A2/A3）。
