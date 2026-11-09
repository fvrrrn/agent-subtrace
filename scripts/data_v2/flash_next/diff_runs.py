"""Show per-case prediction changes between two eval runs.

Usage: python diff_runs.py A.jsonl B.jsonl [--only-changes]
"""
import json
import sys

GOLD = {0: "safe", 1: "unsafe", 2: "uncertain"}


def load(path):
    return {r["key"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    only = "--only-changes" in sys.argv
    fixed = broken = 0
    rows = []
    for key in sorted(a):
        ra, rb = a[key], b.get(key)
        if rb is None or ra["status"] != "ok" or rb["status"] != "ok":
            continue
        if ra["pred"] == rb["pred"] and only:
            continue
        was = ra["pred"] == ra["gold"]
        now = rb["pred"] == rb["gold"]
        if not was and now:
            fixed += 1
            tag = "FIX  "
        elif was and not now:
            broken += 1
            tag = "BREAK"
        else:
            tag = "same "
        rows.append(
            f"{tag} gold={GOLD[ra['gold']]:9s} {GOLD[ra['pred']]:9s} -> {GOLD[rb['pred']]:9s} {key}"
        )
    print("\n".join(rows))
    print(f"\nfixed={fixed} broken={broken} net={fixed - broken:+d}")


if __name__ == "__main__":
    main()
