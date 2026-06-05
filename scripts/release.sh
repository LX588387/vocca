#!/usr/bin/env bash
# 发布辅助脚本：本地跑一遍检查，然后打 tag（推送由人工确认）。
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "用法: scripts/release.sh <version>   例如 scripts/release.sh 0.3.0" >&2
  exit 1
fi

VERSION="$1"
TAG="v${VERSION}"

echo "==> 校验版本号与 vocca/version.py 一致"
grep -q "\"${VERSION}\"" vocca/version.py || {
  echo "vocca/version.py 里的版本号不是 ${VERSION}" >&2
  exit 1
}

echo "==> 运行检查（ruff + mypy + 测试）"
ruff check .
black --check .
mypy
pytest -m "not torch" --cov=vocca --cov-fail-under=85

echo "==> 打 tag ${TAG}"
git tag -a "${TAG}" -m "release ${TAG}"
echo "完成。确认无误后执行： git push origin ${TAG}"
