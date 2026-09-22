import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Laya Blackjack Laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    fetch = commands.add_parser("fetch-model")
    fetch.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/released"))
    fetch.add_argument("--repo")
    fetch.add_argument("--revision")
    package = commands.add_parser("package-model")
    package.add_argument("--source", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--card", type=Path, default=Path("MODEL_CARD.md"))
    package.add_argument("--project", type=Path, default=Path("."))
    package.add_argument("--evidence", type=Path)
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
    evaluate = commands.add_parser("evaluate-policy")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--policy", choices=["laya", "basic", "reference"], default="laya")
    evaluate.add_argument("--source")
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--mode", choices=["fresh", "continuous"], default="fresh")
    evaluate.add_argument("--units", type=int, default=100000)
    evaluate.add_argument("--seed", type=int, default=20260922)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--shard-size", type=int, default=256)
    evaluate.add_argument("--samples", type=int, default=256)
    compare = commands.add_parser("compare-evaluations")
    compare.add_argument("--input", action="append", required=True, help="Policy name=artifact directory")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--candidate", default="candidate")
    audit = commands.add_parser("audit-model")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--source", required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--device", default="cuda:1")
    audit.add_argument("--batch-size", type=int, default=32)
    audit.add_argument("--allow-new-dataset", action="store_true")
    recheck = commands.add_parser("recheck-errors")
    recheck.add_argument("--audit", type=Path, required=True)
    recheck.add_argument("--output", type=Path, required=True)
    recheck.add_argument("--workers", type=int, default=24)
    recheck.add_argument("--samples", type=int, default=10000)
    recheck.add_argument("--repeats", type=int, default=4)
    suite = commands.add_parser("evaluate-suite")
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--policy", choices=["laya", "basic"], default="laya")
    suite.add_argument("--source")
    suite.add_argument("--device", default="cuda:0")
    suite.add_argument("--fresh-units", type=int, default=100000)
    suite.add_argument("--blocks", type=int, default=1000)
    suite.add_argument("--seed", type=int, default=20260922)
    research = commands.add_parser("research")
    research.add_argument("--output", type=Path, required=True)
    research.add_argument("--candidate", required=True)
    research.add_argument("--baseline", required=True)
    research.add_argument("--fresh-units", type=int, default=100000)
    research.add_argument("--blocks", type=int, default=1000)
    research.add_argument("--hours", type=float, default=3)
    research.add_argument("--seed", type=int, default=20260922)
    visitation = commands.add_parser("visitation-pilot", help="Qualify development-only model-visited states")
    visitation.add_argument("--output", type=Path, required=True)
    visitation.add_argument("--source", required=True)
    visitation.add_argument("--workers", type=int, default=24)
    visitation.add_argument("--hours", type=float, default=1)
    visitation.add_argument("--seed", type=int, default=20261002)
    visitation.add_argument("--smoke", action="store_true")
    visit_worker = commands.add_parser("visitation-worker", help="Internal frozen-plan pilot worker")
    visit_worker.add_argument("--root", type=Path, required=True)
    visit_worker.add_argument("--phase", choices=["collect", "label", "audit"], required=True)
    visit_worker.add_argument("--policy", choices=["laya", "mixture"], default="laya")
    visit_worker.add_argument("--device", default="cuda:1")
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
    large.add_argument("--profile", choices=["standard", "composition-v1"], default="standard")
    large.add_argument("--teacher", choices=["monte-carlo", "hybrid-exact-v1"], default="monte-carlo")
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
    large_train.add_argument("--defer-test", action="store_true")
    large_train.add_argument("--general-margin", type=float)
    large_train.add_argument("--objective", choices=["imitation", "cost-sensitive"], default="imitation")
    targeted = commands.add_parser("targeted")
    targeted.add_argument("--output", type=Path, required=True)
    targeted.add_argument("--source", required=True)
    targeted.add_argument("--hours", type=float, default=12)
    targeted.add_argument("--states", type=int, default=65536)
    targeted.add_argument("--selection", type=int, default=4096)
    targeted.add_argument("--calibration", type=int, default=2048)
    targeted.add_argument("--test", type=int, default=8192)
    targeted.add_argument("--workers", type=int, default=24)
    targeted.add_argument("--epochs", type=int, default=3)
    targeted.add_argument("--seed", type=int, default=20260924)
    targeted.add_argument("--fresh-units", type=int, default=100000)
    targeted.add_argument("--blocks", type=int, default=1000)
    targeted.add_argument("--smoke", action="store_true", help="Small execution check, not research evidence")
    targeted.add_argument("--study", choices=["composition", "teacher-cost"], default="composition")
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
    elif command == "fetch-model":
        from .releases import fetch_model

        print(f"Verified model installed at {fetch_model(**args)}")
    elif command == "package-model":
        from .releases import package_model

        result = package_model(**args)
        print(json.dumps({"files": len(result["files"]), "format": result["format"]}))
    elif command == "evaluate-policy":
        from .evaluation import evaluate_policy

        result = evaluate_policy(**args)
        print(json.dumps({"complete": True, "elapsed_seconds": result["elapsed_seconds"]}))
    elif command == "compare-evaluations":
        from .evaluation import compare_evaluations

        inputs = dict(value.split("=", 1) for value in args.pop("input"))
        result = compare_evaluations({k: Path(v) for k, v in inputs.items()}, **args)
        print(json.dumps(result, indent=2))
    elif command == "audit-model":
        from .model_audit import audit_model

        result = audit_model(**args)
        print(json.dumps(result["metrics"], indent=2))
    elif command == "recheck-errors":
        from .model_audit import recheck_errors

        result = recheck_errors(**args)
        print(json.dumps({"cases": len(result["cases"]), "elapsed_seconds": result["elapsed_seconds"]}))
    elif command in ("research", "evaluate-suite"):
        from .research import evaluate_suite, run_research

        (run_research if command == "research" else evaluate_suite)(**args)
    elif command == "targeted":
        from .targeted import run_targeted

        run_targeted(**args)
    elif command in ("visitation-pilot", "visitation-worker"):
        from .visitation import pilot_worker, run_visitation

        (run_visitation if command == "visitation-pilot" else pilot_worker)(**args)
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
