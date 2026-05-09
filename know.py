#!/usr/bin/env python3
"""
know.py — Upsert knowledge entries into the corpus.

Each domain → .janet/knowledge/<domain>.jsonl
Each subdomain → tagged on the entry, not a separate file

Usage:
  python3 know.py add <domain> <subdomain> <id> "<claim>"
  python3 know.py add physics quantum born-rule "The Born rule states..."
  python3 know.py list [domain]
  python3 know.py get <id>
  python3 know.py rm <id>

Writes to .janet/knowledge/ found by walking up from cwd (like git).
Set JANET_DIR to override the search root.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from janet import resolve_janet

ROOT = Path(os.environ.get("JANET_DIR", os.getcwd()))
KB   = resolve_janet(str(ROOT)) / "knowledge"


def _path(domain: str) -> Path:
    KB.mkdir(exist_ok=True)
    return KB / f"{domain}.jsonl"


def _load(domain: str) -> list[dict]:
    p = _path(domain)
    if not p.exists():
        return []
    entries = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def _save(domain: str, entries: list[dict]):
    p = _path(domain)
    p.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")


def _all_domains() -> list[str]:
    return sorted(p.stem for p in KB.glob("*.jsonl"))


def _find(eid: str) -> tuple[str, dict] | None:
    for domain in _all_domains():
        for e in _load(domain):
            if e.get("id") == eid:
                return domain, e
    return None


def cmd_add(domain: str, subdomain: str, eid: str, claim: str,
            confidence: str = "verified", tags: list[str] | None = None,
            source: str = ""):
    """Upsert one entry. Overwrites if id already exists."""
    entries = _load(domain)
    entry = {
        "id":         eid,
        "domain":     domain,
        "subdomain":  subdomain,
        "claim":      claim.strip(),
        "tags":       tags or [domain, subdomain],
        "confidence": confidence,
    }
    if source:
        entry["source"] = source

    # Upsert
    for i, e in enumerate(entries):
        if e.get("id") == eid:
            entries[i] = entry
            _save(domain, entries)
            print(f"updated [{eid}] in {domain}.jsonl")
            return

    entries.append(entry)
    _save(domain, entries)
    print(f"added   [{eid}] to {domain}.jsonl ({len(entries)} entries)")


def cmd_list(domain: str | None = None):
    domains = [domain] if domain else _all_domains()
    for d in domains:
        entries = _load(d)
        if not entries:
            continue
        print(f"\n{d}.jsonl ({len(entries)} entries)")
        for e in entries:
            sub  = e.get("subdomain", "")
            claim = e.get("claim", "")[:80]
            print(f"  [{e['id']}] {sub:<20} {claim}")


def cmd_get(eid: str):
    result = _find(eid)
    if not result:
        print(f"not found: {eid}")
        return
    _, e = result
    print(json.dumps(e, indent=2, ensure_ascii=False))


def cmd_rm(eid: str):
    for domain in _all_domains():
        entries = _load(domain)
        new = [e for e in entries if e.get("id") != eid]
        if len(new) < len(entries):
            _save(domain, new)
            print(f"removed [{eid}] from {domain}.jsonl")
            return
    print(f"not found: {eid}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "add":
        if len(args) < 5:
            print("usage: know.py add <domain> <subdomain> <id> <claim> [confidence] [source]")
            sys.exit(1)
        domain, subdomain, eid, claim = args[1], args[2], args[3], args[4]
        confidence = args[5] if len(args) > 5 else "verified"
        source     = args[6] if len(args) > 6 else ""
        cmd_add(domain, subdomain, eid, claim, confidence, source=source)

    elif cmd == "list":
        cmd_list(args[1] if len(args) > 1 else None)

    elif cmd == "get":
        if len(args) < 2:
            print("usage: know.py get <id>")
            sys.exit(1)
        cmd_get(args[1])

    elif cmd == "rm":
        if len(args) < 2:
            print("usage: know.py rm <id>")
            sys.exit(1)
        cmd_rm(args[1])

    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
