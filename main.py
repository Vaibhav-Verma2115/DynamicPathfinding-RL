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

    exp = sub.add_parser("experiment", help="run an experiment and save raw CSV results")
    exp.add_argument("name", choices=["train_dqn", "e1", "e2", "e3", "e4", "e5", "e6"])
    exp.add_argument("--config", help="experiment YAML (default: configs/experiments/<name>*.yaml)")
    exp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    exp.add_argument("--workers", type=int, default=1, help="parallel worker processes (1 = serial, clean timings)")

    figs = sub.add_parser("figures", help="build summary tables and figures from outputs/results")
    figs.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")

    sub.add_parser("report", help="generate docs/RESULTS.md from the summary tables")

    args = parser.parse_args()
    if args.command == "hardware":
        print(json.dumps(hardware_info(), indent=2))
    elif args.command == "demo":
        from pathrl.visualization.demo import run_demo

        run_demo(load_config(args.config, args.set), steps=args.steps, seed=args.seed)
    elif args.command == "plan":
        from pathrl.visualization.planner_demo import run_planner_demo

        run_planner_demo(load_config(args.config, args.set), seed=args.seed)
    elif args.command == "experiment":
        import importlib
        from pathlib import Path

        modules = {"train_dqn": "train_dqn", "e1": "e1_static_scaling", "e2": "e2_heuristic_quality",
                   "e3": "e3_dynamic_obstacles", "e4": "e4_stochastic", "e5": "e5_generalization",
                   "e6": "e6_convergence"}
        config = args.config or next(Path("configs/experiments").glob(f"{modules[args.name]}*.yaml"))
        module = importlib.import_module(f"pathrl.experiments.{modules[args.name]}")
        module.run(load_config(config, args.set), workers=args.workers)
    elif args.command == "figures":
        from pathrl.experiments.make_figures import run as make_figures

        make_figures(load_config(overrides=args.set))
    elif args.command == "report":
        from pathrl.experiments.make_report import run as make_report

        make_report(load_config())


if __name__ == "__main__":
    main()
