import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Laya Blackjack Laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    generate = commands.add_parser("generate")
    generate.add_argument("--output", type=Path, default=Path("artifacts/data/blackjack"))
    generate.add_argument("--states", type=int, default=500)
    generate.add_argument("--samples", type=int, default=256)
    generate.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train")
    train.add_argument("--dataset", type=Path, default=Path("artifacts/data/blackjack"))
    train.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/blackjack"))
    train.add_argument("--source", default="convaiinnovations/laya")
    train.add_argument("--device", default=None)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=0.0001)
    train.add_argument("--full-model", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    bench.add_argument("--rounds", type=int, default=200)
    bench.add_argument("--seed", type=int, default=12345)
    bench.add_argument("--players", type=int, default=3)
    bench.add_argument("--samples", type=int, default=128)
    bench.add_argument("--model-path")
    replay = commands.add_parser("replay")
    replay.add_argument("path", type=Path)
    large = commands.add_parser("generate-large")
    large.add_argument("--output", type=Path, required=True)
    large.add_argument("--states", type=int, default=100000)
    large.add_argument("--selection", type=int, default=2000)
    large.add_argument("--calibration", type=int, default=2000)
    large.add_argument("--test", type=int, default=5000)
    large.add_argument("--workers", type=int, default=24)
    large.add_argument("--shard-size", type=int, default=128)
    large.add_argument("--samples", type=int, default=1024)
    large.add_argument("--max-samples", type=int, default=4096)
    large.add_argument("--evaluation-samples", type=int, default=4096)
    large.add_argument("--evaluation-max-samples", type=int, default=8192)
    large.add_argument("--seed", type=int, default=20260921)
    large.add_argument("--deadline", type=float)
    large_train = commands.add_parser("train-large")
    large_train.add_argument("--dataset", type=Path, required=True)
    large_train.add_argument("--output", type=Path, required=True)
    large_train.add_argument("--source", required=True)
    large_train.add_argument("--device", default="cuda:0")
    large_train.add_argument("--epochs", type=int, default=3)
    large_train.add_argument("--batch-size", type=int, default=8)
    large_train.add_argument("--learning-rate", type=float, default=0.00001)
    large_train.add_argument("--checkpoint-steps", type=int, default=1000)
    large_train.add_argument("--seed", type=int, default=20260921)
    large_train.add_argument("--deadline", type=float)
    overnight = commands.add_parser("overnight")
    overnight.add_argument("--output", type=Path, required=True)
    overnight.add_argument("--source", required=True)
    overnight.add_argument("--hours", type=float, default=12)
    overnight.add_argument("--states", type=int, default=100000)
    overnight.add_argument("--workers", type=int, default=24)
    overnight.add_argument("--epochs", type=int, default=3)
    overnight.add_argument("--batch-size", type=int, default=8)
    overnight.add_argument("--gpus", default="0,1")
    overnight.add_argument("--selection", type=int, default=2000)
    overnight.add_argument("--calibration", type=int, default=2000)
    overnight.add_argument("--test", type=int, default=5000)
    overnight.add_argument("--benchmark-rounds", type=int, default=2000)
    overnight.add_argument("--seed", type=int, default=20260921)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "serve":
        import uvicorn

        uvicorn.run("blackjack.server:app", **args)
    elif command == "replay":
        from .engine import Game, Rules

        recording = json.loads(args["path"].read_text())
        if recording.get("format") != "laya-blackjack-replay-v1":
            parser.error("Unsupported replay format")
        game = Game(Rules(**recording["rules"]), recording["seed"])
        for action in recording["actions"]:
            game.deal() if action == "deal" else game.step(action)
        if game.observation() != recording["state"] or game.history != recording["history"]:
            raise ValueError("Replay diverged from the recorded session.")
        print(json.dumps({"verified": True, "rounds": game.round, "bankroll": game.bankroll}, indent=2))
    elif command in ("generate-large", "train-large", "overnight"):
        from .experiment_data import generate_large_dataset
        from .experiment_train import train_large
        from .overnight import run_overnight

        fn = {
            "generate-large": generate_large_dataset,
            "train-large": train_large,
            "overnight": run_overnight,
        }[command]
        result = fn(**args)
        # The complete manifest is on disk; avoid flooding logs with thousands of shard receipts.
        print(
            json.dumps(
                {
                    "command": command,
                    "output": str(args["output"]),
                    "complete": True,
                    "summary": result.get("actual_states", result.get("test", result.get("status"))),
                },
                indent=2,
            )
        )
    else:
        from .training import benchmark, generate_dataset
        from .training import train as train_model

        fn = {"generate": generate_dataset, "train": train_model, "benchmark": benchmark}[command]
        print(json.dumps(fn(**args), indent=2))


if __name__ == "__main__":
    main()
