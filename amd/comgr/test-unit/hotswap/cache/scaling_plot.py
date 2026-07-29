#!/usr/bin/env python3
# Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Turn scaling-harness CSV into scaling curves. Reads either binary's CSV
# (dispatched on the `experiment` column). Falls back to ASCII tables if
# matplotlib is unavailable.
#
# Usage:
#   python3 scaling_plot.py results.csv [--out-dir plots]
#   (accepts concatenated CSVs from both binaries)

import csv
import sys
import os
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def ascii_table(title, headers, data_rows):
    print("\n" + title)
    widths = [len(h) for h in headers]
    for row in data_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in data_rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def report_hash_throughput(rows, plt, out_dir):
    rows = [r for r in rows if r.get("experiment") == "hash-throughput"]
    if not rows:
        return
    # ns_per_byte vs source_bytes, one line per hash.
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["hash"]].append((int(r["source_bytes"]), f(r["ns_per_byte"])))
    for h in by_hash:
        by_hash[h].sort()

    if plt:
        plt.figure()
        for h, pts in sorted(by_hash.items()):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=h)
        plt.xscale("log", base=2)
        plt.xlabel("source bytes")
        plt.ylabel("ns / byte")
        plt.title("Hash throughput: ns/byte vs source size")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        p = os.path.join(out_dir, "hash_ns_per_byte.png")
        plt.savefig(p, dpi=120, bbox_inches="tight")
        print("wrote", p)
    else:
        sizes = sorted({int(r["source_bytes"]) for r in rows})
        headers = ["hash"] + [str(s) for s in sizes]
        data = []
        for h, pts in sorted(by_hash.items()):
            m = dict(pts)
            data.append([h] + ["%.4f" % m.get(s, 0.0) for s in sizes])
        ascii_table("ns/byte by hash x source_bytes", headers, data)


def report_cache_scaling(rows, plt, out_dir):
    rows = [r for r in rows if r.get("experiment") == "cache-scaling"]
    if not rows:
        return
    # latency_us vs threads, per (hash, mode, phase). Focus warm (hit path).
    for phase in ("cold", "warm"):
        sub = [r for r in rows if r["phase"] == phase]
        if not sub:
            continue
        series = defaultdict(list)  # (hash,mode) -> [(threads, latency)]
        for r in sub:
            series[(r["hash"], r["mode"])].append(
                (int(r["threads"]), f(r["latency_us"])))
        for k in series:
            series[k].sort()

        if plt:
            plt.figure()
            for (h, mode), pts in sorted(series.items()):
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                plt.plot(xs, ys, marker="o", label=f"{h}/{mode}")
            plt.xlabel("threads")
            plt.ylabel("latency (us)")
            plt.title(f"Cache scaling ({phase}): latency vs threads")
            plt.legend(fontsize=7)
            plt.grid(True, alpha=0.3)
            p = os.path.join(out_dir, f"cache_latency_{phase}.png")
            plt.savefig(p, dpi=120, bbox_inches="tight")
            print("wrote", p)
        else:
            threads = sorted({int(r["threads"]) for r in sub})
            headers = ["hash/mode"] + [str(t) for t in threads]
            data = []
            for (h, mode), pts in sorted(series.items()):
                m = dict(pts)
                data.append([f"{h}/{mode}"] +
                            [str(int(m.get(t, 0))) for t in threads])
            ascii_table(f"latency_us ({phase}) by hash/mode x threads",
                        headers, data)

    # Sanity echo: producer_calls for cold, single-flight should be 1; per-thread == threads.
    cold = [r for r in rows if r["phase"] == "cold"]
    ascii_table(
        "SANITY cold producer_calls (single-flight should be 1, per-thread==threads)",
        ["hash", "mode", "threads", "producer_calls", "coalesced"],
        [[r["hash"], r["mode"], r["threads"], r["producer_calls"], r["coalesced"]]
         for r in sorted(cold, key=lambda x: (x["hash"], x["mode"], int(x["threads"])))],
    )


def main():
    if len(sys.argv) < 2:
        print("usage: scaling_plot.py results.csv [--out-dir DIR]")
        sys.exit(2)
    path = sys.argv[1]
    out_dir = "plots"
    if "--out-dir" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--out-dir") + 1]
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
        print("(matplotlib unavailable -> ASCII tables)")
    rows = load(path)
    report_hash_throughput(rows, plt, out_dir)
    report_cache_scaling(rows, plt, out_dir)


if __name__ == "__main__":
    main()
