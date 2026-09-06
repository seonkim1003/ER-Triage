import argparse

from .data import download
from .model import train
from .replay import replay
from .workload import workload
from .early_warning import early_warning
from .dashboard import dashboard
from .basis import basis
from .contrasts import contrasts
from .sweep import BUDGETS, sweep
from .transfer import SIZES, REPEATS, transfer
from .target import HORIZON


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
    t.add_argument("--target", choices=["persistent", "event"], default="persistent",
                   help="persistent dataset label, or an event-anchored label that is positive only"
                        " within --horizon hours before the onset proxy and drops post-onset hours")
    t.add_argument("--horizon", type=int, default=HORIZON,
                   help="Event-target horizon in hours before the onset proxy")
    r = sub.add_parser("replay")
    r.add_argument("--data", default="data/physionet2019")
    r.add_argument("--run", default="artifacts/baseline")
    r.add_argument("--patient")
    w = sub.add_parser("workload", help="Audit frozen review workload across all held-out patients")
    w.add_argument("--data", default="data/physionet2019")
    w.add_argument("--run", required=True)
    w.add_argument("--out", required=True, help="New directory for workload outputs")
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--draws", type=int, default=1000)
    e = sub.add_parser("early-warning", help="Evaluate patient-level warning timing and repeated alerts")
    e.add_argument("--data", default="data/physionet2019")
    e.add_argument("--run", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--draws", type=int, default=1000)
    b = sub.add_parser("basis", help="Re-score a frozen run on the event-anchored pre-onset basis")
    b.add_argument("--data", default="data/physionet2019")
    b.add_argument("--run", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--horizon", type=int, default=HORIZON,
                   help="Hours before the onset proxy that count as positive")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--draws", type=int, default=1000)
    k = sub.add_parser("contrasts", help="Prespecified subgroup contrasts with Holm correction")
    k.add_argument("--data", default="data/physionet2019")
    k.add_argument("--run", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--seed", type=int, default=42)
    k.add_argument("--draws", type=int, default=1000)
    w2 = sub.add_parser("sweep", help="Coverage against alert burden, thresholds chosen on validation")
    w2.add_argument("--data", default="data/physionet2019")
    w2.add_argument("--run", required=True)
    w2.add_argument("--against", help="Second run to interpolate to matched burden")
    w2.add_argument("--out", required=True)
    w2.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS),
                    help="Comma-separated validation alert-hour budgets per 100")
    w2.add_argument("--seed", type=int, default=42)
    w2.add_argument("--draws", type=int, default=400)
    x = sub.add_parser("transfer", help="How much target-site data recalibration needs to transfer")
    x.add_argument("--data", default="data/physionet2019")
    x.add_argument("--run", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--sizes", default=",".join(str(n) for n in SIZES),
                   help="Comma-separated target-site patient counts to fit on")
    x.add_argument("--repeats", type=int, default=REPEATS)
    x.add_argument("--seed", type=int, default=42)
    v = sub.add_parser("dashboard", help="Serve a local read-only visual patient replay")
    v.add_argument("--data", default="data/physionet2019")
    v.add_argument("--run", required=True)
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--evaluation", help="Matching early_warning.json to display cohort findings")
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
            if args.horizon < 1:
                parser.error("--horizon must be a positive whole number of hours")
            train(args.data, args.out, args.limit, args.seed, args.split, args.draws,
                  args.threshold, args.alert_budget, args.select, args.target, args.horizon)
        elif args.command == "workload":
            if args.draws < 1:
                parser.error("--draws must be positive")
            workload(args.data, args.run, args.out, seed=args.seed, draws=args.draws)
        elif args.command == "early-warning":
            if args.draws < 1:
                parser.error("--draws must be positive")
            early_warning(args.data, args.run, args.out, args.seed, args.draws)
        elif args.command == "basis":
            if args.draws < 1:
                parser.error("--draws must be positive")
            if args.horizon < 1:
                parser.error("--horizon must be a positive whole number of hours")
            basis(args.data, args.run, args.out, args.horizon, args.seed, args.draws)
        elif args.command == "contrasts":
            if args.draws < 1:
                parser.error("--draws must be positive")
            contrasts(args.data, args.run, args.out, args.seed, args.draws)
        elif args.command == "sweep":
            if args.draws < 1:
                parser.error("--draws must be positive")
            try:
                budgets = tuple(float(b) for b in args.budgets.split(","))
            except ValueError:
                parser.error("--budgets must be comma-separated numbers")
            if not budgets or not all(0 < b <= 100 for b in budgets):
                parser.error("--budgets must lie within (0, 100] alert hours per 100")
            sweep(args.data, args.run, args.out, args.against, budgets, args.seed, args.draws)
        elif args.command == "transfer":
            if args.repeats < 1:
                parser.error("--repeats must be positive")
            try:
                sizes = tuple(int(n) for n in args.sizes.split(","))
            except ValueError:
                parser.error("--sizes must be comma-separated whole numbers")
            if not sizes or not all(n > 0 for n in sizes):
                parser.error("--sizes must be positive patient counts")
            transfer(args.data, args.run, args.out, sizes, args.repeats, args.seed)
        elif args.command == "dashboard":
            if not 1 <= args.port <= 65535:
                parser.error("--port must be within 1..65535")
            dashboard(args.data, args.run, args.port, args.evaluation)
        else:
            replay(args.data, args.run, args.patient)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
