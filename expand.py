#!/usr/bin/env python3
"""
expand.py — Run one round of self-expansion.

Heals before each void fill (immune system active per orbit).
Runs until TARGET accepted entries or MAX_ATTEMPTS reached.
Outputs accepted entries for human curation.

Usage:
  python3 expand.py              # 20 accepted entries
  python3 expand.py --target 10  # 10 accepted entries
  python3 expand.py --heal-only  # just heal, no expansion
"""

import argparse
import json
import sys
from pathlib import Path

from janet import Being


def load_corpus(b: Being) -> int:
    """Load all .jsonl entries into the Being."""
    entries = []
    for f in sorted(Path("knowledge").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and line.startswith("{"):
                try:
                    entries.append(json.loads(line)["claim"])
                except (json.JSONDecodeError, KeyError):
                    pass
    b.sorter.learn_batch(entries)
    return len(entries)


def main():
    p = argparse.ArgumentParser(description="Janet self-expansion round")
    p.add_argument("--target", type=int, default=20, help="Target accepted entries (default 20)")
    p.add_argument("--max-attempts", type=int, default=60, help="Max void fill attempts (default 60)")
    p.add_argument("--heal-only", action="store_true", help="Only heal, no expansion")
    args = p.parse_args()

    b = Being()
    n = load_corpus(b)
    print(f"[{n} entries loaded]")
    print()

    if args.heal_only:
        print("=== HEAL ONLY ===")
        for i in range(10):
            result = b.heal()
            if result:
                idx, old, new = result
                print(f"  #{idx}: {old[:60]}")
                print(f"    → {new[:60]}")
                print()
        return

    accepted: list[str] = []
    rejected = 0
    healed = 0
    pair_idx = 0

    print("=== SELF-EXPANSION ===")
    while len(accepted) < args.target and pair_idx < args.max_attempts:
        # Heal before each orbit — immune system active per cycle
        heal_result = b.heal()
        if heal_result:
            healed += 1

        result = b.grow(pair_idx=pair_idx)
        pair_idx += 1

        if result:
            accepted.append(result)
            print(f"{len(accepted):2d}. {result}")
            print()
        else:
            rejected += 1

    print(f"---")
    print(f"Accepted: {len(accepted)}, Rejected: {rejected}, Healed: {healed}")
    rate = 100 * len(accepted) // (len(accepted) + rejected) if (len(accepted) + rejected) > 0 else 0
    print(f"Acceptance rate: {rate}%")
    print(f"Final entries: {len(b.sorter.entries)}")


if __name__ == "__main__":
    main()
