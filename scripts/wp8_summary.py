#!/usr/bin/env python3
"""Collect every results/wp8/<case>/comparison.json into one table.

    python scripts/wp8_summary.py            # writes results/wp8/comparison_summary.csv
    python scripts/wp8_summary.py --print    # also prints it

Distances and ranks come from the per-case measurements. The joint criterion
combines equality of the first near-native cluster ranks with a 1.6 A cutoff
on the distance from the leading balanced model to its nearest JuPIPER centre.
All threshold comparisons use unrounded values.

Stdlib only.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "wp8"
OUT = RES / "comparison_summary.csv"
FLOOR = 1.6          # estimated joint discretisation floor (angstrom)
HIT = 10.0           # the campaign's near-native threshold, bench/benchmark51.jl


def band(x):
    if x is None:
        return ""
    return "A" if x <= 2.5 else "B" if x <= 5.0 else "C"


def summarise(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    srv = d["server_models"]
    ours = d["our_centres"]
    bal = [m for m in srv if m["set"] == "balanced"]
    bal.sort(key=lambda m: m["cluster"])
    if not bal:
        raise SystemExit(f"{path}: no balanced models")

    # Their top-ranked (largest) balanced cluster, and ours.
    t1 = bal[0]
    o1 = ours[0]
    # Their near-native cluster rank, by the same 10 A rule the campaign uses on
    # our side (the centre's ligand-Ca RMSD to the native).
    their_nn_rank = next((m["cluster"] + 1 for m in bal if m["lrmsd_native"] <= HIT), None)
    our_nn_rank = d.get("our_near_native_cluster_rank")

    nn_top = [m["nn_topk"] for m in bal]
    conv = [c["nn_server"] for c in ours]

    def q(v):
        s = sorted(v)
        return s[0], s[len(s) // 2], s[-1]

    same_rank = their_nn_rank is not None and their_nn_rank == our_nn_rank
    within_floor = t1["nn_centre"] <= FLOOR
    return {
        "case": d["case"],
        "job": d["job"],
        "nrot": d["nrot"],
        "search_floor_ca": round(d["search_floor_ca"], 4),
        "their_clusters_balanced": len(bal),
        "our_clusters": d["our_clusters"],
        "their_top_lrmsd_native": round(t1["lrmsd_native"], 2),
        "our_top_lrmsd_native": round(o1["lrmsd_native"], 2),
        "their_nn_cluster_rank": their_nn_rank,
        "our_nn_cluster_rank": our_nn_rank,
        "same_rank_band": same_rank,
        "top_centre_to_centre": round(t1["nn_centre"], 2),
        "top_centre_nearest_our_centre_rank": t1["nn_centre_rank"],
        "pose_within_2A": t1["nn_topk"] <= 2.0,
        "top_centre_to_our_top1000": round(t1["nn_topk"], 2),
        "top_centre_our_rank_of_that_pose": t1["nn_topk_rank"],
        "within_floor": within_floor,
        "joint_criterion": same_rank and within_floor,
        "band_top_centre": band(t1["nn_centre"]),
        "balanced_to_top1000_min": round(q(nn_top)[0], 2),
        "balanced_to_top1000_med": round(q(nn_top)[1], 2),
        "balanced_to_top1000_max": round(q(nn_top)[2], 2),
        "balanced_within_1p6": sum(1 for x in nn_top if x <= FLOOR),
        "balanced_within_2p5": sum(1 for x in nn_top if x <= 2.5),
        "our_centres_to_server_min": round(q(conv)[0], 2),
        "our_centres_to_server_med": round(q(conv)[1], 2),
        "our_centres_to_server_max": round(q(conv)[2], 2),
        "their_best_any_set_lrmsd_native": round(min(m["lrmsd_native"] for m in srv), 2),
        "their_best_any_set": min(srv, key=lambda m: m["lrmsd_native"])["set"],
        "our_best_pose_lrmsd_native": round(d["our_best_rmsd_ca"], 2),
        "rec_fit_max": max(m["rec_fit_ca"] for m in srv),
    }


def main(argv):
    rows = []
    for p in sorted(RES.glob("*/comparison.json")):
        rows.append(summarise(p))
    if not rows:
        raise SystemExit(f"no comparison.json under {RES}")
    cols = list(rows[0].keys())
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# WP8 external comparison -- one row per case, derived from\n")
        f.write("# results/wp8/<case>/comparison.json (including derived aggregates and flags).\n")
        f.write("# Metric: ligand C-alpha displacement in the receptor-fixed frame, no\n")
        f.write(f"# re-superposition. floor = {FLOOR} A (estimated);\n")
        f.write(f"# near-native = centre within {HIT} A of the native, both sides.\n")
        f.write("# top_centre_to_centre: leading balanced ClusPro model to nearest JuPIPER centre.\n")
        f.write("# band_top_centre applies the A/B/C bands (cutoffs 2.5 and 5 A)\n")
        f.write("# to top_centre_to_centre ONLY; they are a reading aid, not the criterion.\n")
        f.write("# The pre-registered criterion is same_rank_band AND within_floor.\n")
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} cases)")
    if "--print" in argv:
        keys = ["case", "their_nn_cluster_rank", "our_nn_cluster_rank", "same_rank_band",
                "top_centre_to_centre", "within_floor", "joint_criterion", "band_top_centre",
                "top_centre_to_our_top1000", "top_centre_our_rank_of_that_pose",
                "balanced_within_1p6", "their_best_any_set_lrmsd_native",
                "our_best_pose_lrmsd_native"]
        print("  ".join(f"{k}" for k in keys))
        for r in rows:
            print("  ".join(str(r[k]) for k in keys))


if __name__ == "__main__":
    main(sys.argv[1:])
