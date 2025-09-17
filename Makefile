.PHONY: help install lint format type test check cov clean demo

help:
	@echo "install  安装开发依赖 (editable + dev)"
	@echo "lint     ruff 检查"
	@echo "format   black + ruff --fix 就地格式化"
	@echo "type     mypy 类型检查"
	@echo "test     运行测试（不含 torch 标记）"
	@echo "cov      测试并输出覆盖率"
	@echo "check    lint + type + test 一把梭"
	@echo "demo     跑一个最小生成示例"

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff check --fix .
	black .

type:
	mypy

test:
	pytest -m "not torch"

cov:
	pytest -m "not torch" --cov=vocca --cov-report=term-missing --cov-fail-under=85

check: lint type cov

demo:
	python examples/quickstart.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
