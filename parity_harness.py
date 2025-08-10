import os
import sys
import argparse
import subprocess
import pandas as pd

SEG_CSV = "results/parity/segmented_trades.csv"
TST_CSV = "results/parity/tester_trades.csv"
SEG_CFG = "results/parity/segmented_config.json"
TST_CFG = "results/parity/tester_config.json"


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def summarize(df: pd.DataFrame) -> tuple[int, float, float]:
    trades = len(df)
    wr = float((df["r"] > 0).mean() * 100.0) if trades else 0.0
    total_r = float(df["r"].sum()) if trades else 0.0
    return trades, wr, total_r


def main() -> int:
    parser = argparse.ArgumentParser(description="Parity harness for ETH sessions")
    parser.add_argument("--session", default="London", help="Asia|London|NewYork")
    args = parser.parse_args()

    os.makedirs("results/parity", exist_ok=True)

    # Run segmented backtester
    rc1 = run([
        sys.executable,
        "session_segmented_backtester.py",
        "--session",
        args.session,
        "--lookback_hours",
        "1500",
        "--export-trades",
        SEG_CSV,
        "--dump-config",
        SEG_CFG,
    ])
    if rc1 != 0:
        print("segmented backtester failed", file=sys.stderr)
        return rc1

    # Run speed tester
    rc2 = run([
        sys.executable,
        "speed_tester.py",
        "ETH/USDT",
        "--session",
        args.session,
        "--lookback_hours",
        "1500",
        "--export-trades",
        TST_CSV,
        "--dump-config",
        TST_CFG,
    ])
    if rc2 != 0:
        print("speed tester failed", file=sys.stderr)
        return rc2

    if not (os.path.exists(SEG_CSV) and os.path.exists(TST_CSV)):
        print("missing CSV outputs", file=sys.stderr)
        return 2

    seg = pd.read_csv(SEG_CSV)
    tst = pd.read_csv(TST_CSV)

    seg_stats = summarize(seg)
    tst_stats = summarize(tst)

    print("segmented:", seg_stats)
    print("tester:", tst_stats)

    # Match trades by timestamp_entry + side
    on_cols = ["timestamp_entry", "side"]
    if not all(c in seg.columns for c in on_cols) or not all(c in tst.columns for c in on_cols):
        print("diagnostic columns missing for matching", file=sys.stderr)
        return 3

    merged = pd.merge(seg, tst, on=on_cols, suffixes=("_seg", "_tst"))
    if len(merged) > 0:
        avg_abs_delta = float((merged["r_seg"] - merged["r_tst"]).abs().mean())
    else:
        avg_abs_delta = float("nan")

    # Report unmatched heads
    seg_keys = set(tuple(r) for r in seg[on_cols].itertuples(index=False, name=None))
    tst_keys = set(tuple(r) for r in tst[on_cols].itertuples(index=False, name=None))
    only_seg = list(seg_keys - tst_keys)[:5]
    only_tst = list(tst_keys - seg_keys)[:5]

    print("avg_abs_delta_R:", avg_abs_delta)
    print("unmatched_in_segmented (first 5):", only_seg)
    print("unmatched_in_tester (first 5):", only_tst)

    # Thresholds
    delta_trades = abs(seg_stats[0] - tst_stats[0])
    delta_wr = abs(seg_stats[1] - tst_stats[1])
    delta_total_r = abs(seg_stats[2] - tst_stats[2])

    ok = (delta_trades <= 1) and (delta_wr <= 2.0) and (delta_total_r <= 1.0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


