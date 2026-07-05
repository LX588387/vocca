"""从配置初始化一套（随机）权重并存成 .npz。

真实项目里这一步会由训练侧产出；这里提供一个可跑的最小版本，方便验证
``load_weights`` 的存取链路，也作为导出脚本的模板。

运行::

    python scripts/export_weights.py --config configs/default.yaml --out acoustic.npz
"""

from __future__ import annotations

import argparse

from vocca.config import VoccaConfig
from vocca.model import init_weights, save_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化并导出声学模型权重")
    parser.add_argument("--config", default=None, help="VoccaConfig 的 YAML 路径（可选）")
    parser.add_argument("--out", default="acoustic.npz", help="输出 .npz 路径")
    parser.add_argument("--seed", type=int, default=0, help="初始化随机种子")
    args = parser.parse_args()

    cfg = VoccaConfig.from_yaml(args.config) if args.config else VoccaConfig()
    cfg.validate()
    weights = init_weights(cfg.model, seed=args.seed)
    save_weights(args.out, weights)
    print(f"已导出 {weights.num_parameters():,} 个参数 → {args.out}")


if __name__ == "__main__":
    main()
