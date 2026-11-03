"""Command-line entry point. Subcommands are added as stages are completed.

Examples:
    python main.py hardware
    python main.py demo --config configs/dynamic.yaml --set env.grid_size=20 --steps 80
    python main.py plan --set env.grid_size=30 --seed 1
"""

from __future__ import annotations

import argparse
import json

from pathrl.utils.config import load_config
from pathrl.utils.hardware import hardware_info


def main() -> None:
    parser = argparse.ArgumentParser(description="DynamicPathfinding-RL")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hardware", help="print the hardware/software snapshot")

    demo = sub.add_parser("demo", help="run a random policy in the GridWorld and save a figure")
    demo.add_argument("--config", help="experiment YAML overriding configs/default.yaml")
    demo.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. env.grid_size=20")
    demo.add_argument("--steps", type=int, default=60)
    demo.add_argument("--seed", type=int)

    plan = sub.add_parser("plan", help="compare BFS, Dijkstra and A* on one map and save a figure")
    plan.add_argument("--config", help="experiment YAML overriding configs/default.yaml")
    plan.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. env.grid_size=30")
    plan.add_argument("--seed", type=int)

    args = parser.parse_args()
    if args.command == "hardware":
        print(json.dumps(hardware_info(), indent=2))
    elif args.command == "demo":
        from pathrl.visualization.demo import run_demo

        run_demo(load_config(args.config, args.set), steps=args.steps, seed=args.seed)
    elif args.command == "plan":
        from pathrl.visualization.planner_demo import run_planner_demo

        run_planner_demo(load_config(args.config, args.set), seed=args.seed)


if __name__ == "__main__":
    main()
