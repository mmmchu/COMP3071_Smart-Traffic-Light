# run_experiments.py
# Usage:
# python run_experiments.py --agent ql
# python run_experiments.py --agent ppo
# python run_experiments.py --agent fixed

import argparse
import csv
import os
from ql_test import test_ql
from ppo_test import test_ppo
from fixed_test import test_fixed

def save_results(agent_name, metrics):
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", "experiment_results.csv")
    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Agent", "Avg Waiting Time", "Avg Queue Length", "Throughput"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Agent": agent_name,
            "Avg Waiting Time": metrics["waiting_time"],
            "Avg Queue Length": metrics["queue_length"],
            "Throughput": metrics["throughput"],
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["ql", "ppo", "fixed"], required=True)
    args = parser.parse_args()

    if args.agent == "ql":
        metrics = test_ql()
    elif args.agent == "ppo":
        metrics = test_ppo()
    elif args.agent == "fixed":
        metrics = test_fixed()

    save_results(args.agent, metrics)
    print(f"[\u2705] Results saved for agent: {args.agent}")
