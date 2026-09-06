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
            train(args.data, args.out, args.limit, args.seed)
        else:
            replay(args.data, args.run, args.patient)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
