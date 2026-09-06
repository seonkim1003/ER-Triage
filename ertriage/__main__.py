import argparse

from .data import download
from .model import train
from .replay import replay


def main():
    parser = argparse.ArgumentParser(description="Local ICU research prototype. Not ER validation.")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--data", default="data/physionet2019")
    d.add_argument("--limit", type=int, default=2000, help="Uniform random subset; 0 means all")
    d.add_argument("--seed", type=int, default=42)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/physionet2019")
    t.add_argument("--out", default="artifacts/baseline")
    t.add_argument("--limit", type=int, default=2000, help="Random patient subset; 0 means all")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--split", choices=["random", "site"], default="random",
                   help="random patient split, or hold out an entire source site")
    t.add_argument("--draws", type=int, default=1000, help="Patient bootstrap resamples for held-out intervals")
    t.add_argument("--threshold", choices=["f1", "utility", "budget"], default="budget",
                   help="Validation threshold rule: an alert budget (default), hourly F1, or official utility")
    t.add_argument("--alert-budget", type=float, default=2.,
                   help="Alert hours per 100 allowed by the budget threshold rule")
    t.add_argument("--select", choices=["ap", "stable"], default="stable",
                   help="Model selection: validation average precision, or step back to the simplest"
                        " model a paired validation bootstrap cannot distinguish from the leader")
    r = sub.add_parser("replay")
    r.add_argument("--data", default="data/physionet2019")
    r.add_argument("--run", default="artifacts/baseline")
    r.add_argument("--patient")
    args = parser.parse_args()
    try:
        if args.command == "download":
            if args.limit < 0:
                parser.error("--limit must be nonnegative")
            download(args.data, args.limit, args.seed)
        elif args.command == "train":
            if args.limit < 0:
                parser.error("--limit must be nonnegative")
            if args.draws < 1:
                parser.error("--draws must be positive")
            if not 0 < args.alert_budget <= 100:
                parser.error("--alert-budget must be within (0, 100] alert hours per 100")
            train(args.data, args.out, args.limit, args.seed, args.split, args.draws,
                  args.threshold, args.alert_budget, args.select)
        else:
            replay(args.data, args.run, args.patient)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
