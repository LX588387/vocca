# 贡献指南

感谢你对 vocca 的兴趣！下面是参与开发的一些约定。

## 开发环境

```bash
git clone https://github.com/LX588387/vocca
cd vocca
python -m pip install -e ".[dev]"
pre-commit install
```

核心只依赖 numpy 与 pyyaml；torch 是可选的（`pip install -e ".[dev,torch]"`）。

## 本地检查

提交前请确保以下命令全部通过（与 CI 一致）：

```bash
ruff check .
black --check .
mypy
pytest --cov=vocca --cov-fail-under=85
```

或直接：

```bash
make check
```

torch 相关测试默认跳过，装了 torch 后可单独运行：

```bash
pytest -m torch
```

## 代码风格

- 遵循 `ruff` + `black`（行宽 100）；公开函数写类型注解与中文 docstring。
- 数值实现优先保证**可复现**：涉及随机的地方一律接受 `seed` / `rng`。
- 新增组件（编解码器等）请通过 `Registry` 注册，并补上对应单测。
- 提交信息推荐 [Conventional Commits](https://www.conventionalcommits.org/) 风格，但不强制。

## 提交 PR

1. 从 `main` 切分支，保持每个 PR 聚焦一件事。
2. 补充测试，保证覆盖率不下降。
3. 面向用户的改动记得更新 `CHANGELOG.md` 的 `Unreleased` 段。
4. CI 全绿后再请求 review。

## 报告问题

用 issue 模板描述复现步骤、期望行为与实际行为；涉及音频的问题请附上配置
（采样率、`frame_size` 等）。
