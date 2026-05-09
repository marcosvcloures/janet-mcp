#!/usr/bin/env python3
"""
mcp.py — Janet MCP server. Pure geometry engine over .jsonl knowledge files.

Source of truth: .janet/{center}/*.jsonl  (one JSON object per line)
Format per entry: {"id": "...", "domain": "...", "claim": "...", "tags": [...],
                   "confidence": "verified|ground-truth|candidate", "source": "..."}

Janet reads .jsonl files at startup, builds a matrix in memory, serves queries.
No geometry is ever written to disk. The file system IS the corpus.

Folder convention:
  .janet/knowledge/  — default semantic center
  .janet/{name}/     — any custom center

"""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from janet import (
    Sorter, particle, normalize, shannon_entropy,
    DEGREES, DIMS, MAX_EMBED_VAL, ORBIT_STEPS, IDEAL_CORPUS,
    resolve_janet,
    _particle_cache,
)

# Shift for Gram-Schmidt rejection — derived from DEGREES, prevents int64 overflow
def _get_S() -> int:
    return (63 - DEGREES) // 2


# ── Embedding ─────────────────────────────────────────────────────────────

def _generate_from_edge(state: np.ndarray, max_steps: int = 8) -> str:
    """Generate words from a Vec using the particle vocabulary.

    Sequential rejection: find best-aligned word, erase its direction, repeat.
    Filters to alphabetic tokens len > 3 — no punctuation/number noise.
    Pure integer arithmetic throughout.
    """
    vocab = {k: v for k, v in _particle_cache.items() if k.isalpha() and len(k) > 3}
    if not vocab:
        return "(vocabulary empty)"

    words  = list(vocab.keys())
    pvecs  = np.stack(list(vocab.values()), axis=0).astype(np.int64)
    orig   = state.astype(np.int64).copy()
    dims   = len(state)
    traj   = np.zeros(dims, dtype=np.int64)
    result: list[str] = []
    seen:   set[str]  = set()

    for _ in range(max_steps):
        amps = pvecs @ state.astype(np.int64)
        best = int(np.argmax(amps))
        if amps[best] <= 0:
            break
        word = words[best]
        if word in seen:
            break
        seen.add(word)
        result.append(word)
        traj = traj + pvecs[best]
        m    = int(np.max(np.abs(traj)))
        if m > MAX_EMBED_VAL:
            traj = traj * MAX_EMBED_VAL // m
        dd = int(np.dot(traj, traj))
        if dd == 0:
            break
        dv    = int(np.dot(orig, traj))
        state = (orig * (dd >> _get_S()) - traj * (dv >> _get_S())).astype(np.int32)
        m     = int(np.max(np.abs(state)))
        if m > MAX_EMBED_VAL:
            state = state * MAX_EMBED_VAL // m
        if m == 0:
            break

    return " ".join(result) if result else "(edge unreachable)"


def entry_text(e: dict) -> str:
    """Format entry for embedding. Route only — domain is filesystem metadata.

    Hierarchical routing lives INSIDE the route field:
      subdomain (first tokens) → coarse routing (which cluster)
      route tokens             → fine-grained discrimination

    Domain is the .jsonl filename, not part of the geometry.
    This prevents a single domain token from dominating all dot products.
    """
    return e.get("route") or e.get("claim", "")


# ── Knowledge store ───────────────────────────────────────────────────────

class KnowledgeStore:
    """All knowledge in memory. Source of truth: .jsonl files.

    Each .jsonl file in the center directory contributes entries.
    A single Sorter encodes all entries for retrieval with confidence signal.
    """

    def __init__(self, center_dir: Path):
        self.dir         = center_dir
        self.entries:    list[dict]        = []
        self.texts:      list[str]         = []
        self.sorter      = Sorter()
        self.matrix      = None
        self._anti_grav  = None
        self._miss_vec   = None
        self.reload()

    def reload(self):
        """Rebuild corpus from all .jsonl files in the center directory."""
        entries: list[dict] = []
        if self.dir.exists():
            for f in sorted(self.dir.rglob("*.jsonl")):
                try:
                    for line in f.read_text(errors="replace").splitlines():
                        line = line.strip()
                        if line:
                            e = json.loads(line)
                            if isinstance(e, dict) and e.get("claim"):
                                entries.append(e)
                except Exception:
                    pass

        self.entries = entries
        self.texts   = [entry_text(e) for e in entries]
        self.sorter  = Sorter()
        if self.texts:
            self.sorter.learn_batch(self.texts)
        self.matrix = self.sorter._matrix

    # ── Query ─────────────────────────────────────────────────────────────

    def query(self, q: str) -> str:
        """Query corpus. Returns the full orbit — not just the destination.

        The orbit is the answer:
          path:       associative chain from query to attractor
          cost:       relaxation time (confidence inverse)
          amplitude:  how inside the sphere the query is
          converged:  fixed point reached vs ambiguity
        """
        if not self.entries:
            return ""

        answer, vec, cost = self.sorter.orbit_with_cost(q)
        confidence_pct = max(0, (ORBIT_STEPS - cost) * 100 // ORBIT_STEPS)

        if not answer:
            return "(outside sphere)"

        # Get full orbit for path
        o = self.sorter.orbit(q)

        # Low confidence: generate coherent response from corpus geometry
        if confidence_pct < 20:
            state = self.sorter.encode(q)
            generated = self.sorter.generate_coherent(state)
            if generated:
                return (f"[generated] {generated}\n"
                        f"orbit: cost={o['cost']}, amplitude={o['amplitude']}, "
                        f"converged={o['converged']}, steps={o['steps']}")

        # Find the matching entry
        e = None
        for entry in self.entries:
            if entry_text(entry) == answer or entry.get("claim", "") in answer:
                e = entry
                break
        if e is None:
            M = self.matrix.astype(np.int64)
            amps = M @ vec.astype(np.int64)
            idx = int(np.argmax(amps))
            e = self.entries[idx]

        # Build orbit-centric response
        out = f"[{e.get('id','')}] {e.get('claim', '')}"
        out += f"\norbit: cost={o['cost']}, amplitude={o['amplitude']}, converged={o['converged']}, steps={o['steps']}"
        if len(o['path']) > 1:
            out += f"\npath: {' → '.join(p[:40] for p in o['path'])}"
        if e.get("source"):
            out += f"\nsource: {e['source']}"

        return out

    def reset_session(self) -> str:
        self._anti_grav = None
        self._miss_vec  = None
        return "session state cleared"

    # ── Ingestion ─────────────────────────────────────────────────────────

    def add(self, domain: str, entry: dict) -> str:
        """Append entry to {center}/{domain}.jsonl and reload."""
        path = self.dir / f"{domain}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        claim      = entry.get("claim", "")
        confidence = entry.get("confidence", "")
        if confidence not in ("verified", "ground-truth") and self.reject(claim):
            return (f"rejected — entry appears foreign to corpus geometry. "
                    f"Set confidence='verified' to override.")
        with open(path, "a") as f:
            f.write(json.dumps({k: v for k, v in entry.items() if v},
                               ensure_ascii=False) + "\n")
        self.reload()
        return f"added [{entry.get('id','')}] to {domain}.jsonl ({len(self.entries)} entries)"

    # ── Biological operations ─────────────────────────────────────────────

    def reject(self, text: str) -> bool:
        """Immune system: True = foreign, False = fits corpus.

        Threshold: max amplitude must reach self_amp // ORBIT_STEPS.
        Same threshold as the confident-sphere query boundary.
        """
        if self.matrix is None or not self.entries:
            return False
        qvec     = self.sorter.encode(text)
        self_amp = int(np.dot(qvec.astype(np.int64), qvec.astype(np.int64)))
        if self_amp == 0:
            return True
        max_amp  = int((self.matrix.astype(np.int64) @ qvec.astype(np.int64)).max())
        return max_amp < self_amp // ORBIT_STEPS

    def suggest_route(self, claim: str) -> str:
        """Suggest routing tags for a new entry based on corpus geometry.

        Finds the most aligned existing entries and extracts their shared
        tokens as suggested tags. The geometry tells you which tokens
        will create the strongest bridges to existing knowledge.

        Returns a route string the LLM can use or refine.
        """
        if not self.entries or self.matrix is None:
            return claim.lower()

        # Encode the claim and find top-3 aligned entries
        qvec = self.sorter.encode(claim)
        amps = self.matrix.astype(np.int64) @ qvec.astype(np.int64)
        top_indices = np.argsort(amps)[-3:][::-1]

        # Extract tokens from top entries' route texts
        neighbor_tokens: dict[str, int] = {}
        for idx in top_indices:
            if amps[idx] <= 0:
                continue
            route_text = self.texts[int(idx)]
            for tok in route_text.lower().split():
                if len(tok) > 3 and tok.isalpha():
                    neighbor_tokens[tok] = neighbor_tokens.get(tok, 0) + 1

        # Tokens from the claim itself
        claim_tokens = [tok for tok in claim.lower().split()
                        if len(tok) > 3 and tok.isalpha()]

        # Suggested route: claim tokens + shared neighbor tokens (bridges)
        bridge_tokens = [tok for tok, count in sorted(neighbor_tokens.items(),
                         key=lambda x: -x[1]) if count >= 2 and tok not in claim_tokens][:5]

        route_parts = claim_tokens[:8] + bridge_tokens
        return " ".join(dict.fromkeys(route_parts))  # deduplicate, preserve order

    def improve_routes(self) -> list[dict]:
        """Find entries with weak connections AND ubiquitous tokens to remove.

        Returns list of improvements:
        - Weak entries: connectivity < 30%, suggests better routes
        - Ubiquitous tokens: appear in >50% of entries, should be removed from routes
          (they don't discriminate — like having 'the' in every route)

        This is automatic self-heal: the geometry detects what hurts routing.
        """
        if not self.entries or self.matrix is None or len(self.entries) < 4:
            return []

        M = self.matrix.astype(np.int64)
        gram = M @ M.T
        np.fill_diagonal(gram, 0)
        max_neighbor = gram.max(axis=1)
        self_amps = np.array([int(np.dot(M[i], M[i])) for i in range(len(M))])

        improvements = []

        # Detect ubiquitous tokens (>50% of entries — noise, not signal)
        token_df: dict[str, int] = {}
        for text in self.texts:
            for tok in set(text.lower().split()):
                if len(tok) > 2:
                    token_df[tok] = token_df.get(tok, 0) + 1

        n = len(self.entries)
        ubiquitous = [tok for tok, df in token_df.items() if df > n // 2 and n > 4]
        if ubiquitous:
            improvements.append({
                "type": "ubiquitous_tokens",
                "tokens": ubiquitous,
                "reason": f"These tokens appear in >50% of entries and don't discriminate. Remove from routes.",
                "affected_entries": n,
            })

        # Detect weak entries
        for i, e in enumerate(self.entries):
            if self_amps[i] == 0:
                continue
            ratio = float(max_neighbor[i]) / float(self_amps[i])
            if ratio < 0.3:
                suggested = self.suggest_route(e.get("claim", ""))
                current = e.get("route", entry_text(e))
                improvements.append({
                    "type": "weak_entry",
                    "id": e.get("id", f"entry_{i}"),
                    "current_route": current,
                    "suggested_route": suggested,
                    "connectivity": round(ratio, 3),
                })

        return improvements

    def hunger(self) -> dict:
        """How hungry is the corpus? Integer score 0–100."""
        if self.matrix is None or not self.entries:
            return {"score": 100, "isolated": 0, "total": 0, "message": "empty corpus"}
        M            = self.matrix.astype(np.int64)
        gram         = M @ M.T
        np.fill_diagonal(gram, np.iinfo(np.int64).min)
        max_neighbor = gram.max(axis=1)
        self_amps    = np.array([int(np.dot(M[i], M[i])) for i in range(len(M))])
        threshold    = self_amps // ORBIT_STEPS
        n_isolated   = int((max_neighbor < threshold).sum())
        score        = n_isolated * 100 // max(1, len(self.entries))
        msg = (
            "starving — major voids"     if score > 50 else
            "hungry — several isolated"  if score > 25 else
            "peckish — minor gaps"       if score > 10 else
            "well-fed"
        )
        return {"score": score, "isolated": n_isolated,
                "total": len(self.entries), "message": msg}

    def health(self) -> dict:
        """Full self-assessment with maintenance recommendations."""
        if self.matrix is None or not self.entries:
            return {"status": "empty", "action": "add"}
        h        = self.hunger()
        M        = self.matrix.astype(np.int64)
        amps     = np.array([int(np.dot(M[i], M[i])) for i in range(len(M))])
        weakest  = int(amps.min())
        median_a = int(np.partition(amps, len(amps) // 2)[len(amps) // 2])
        quality  = int((amps > 0).sum()) * 100 // max(1, len(self.entries))

        # Compute self_cost for maintenance signals
        sc = self.sorter.self_cost() if len(self.entries) >= 4 else 1.0

        # Determine action and maintenance needs
        maintenance = []
        if sc > 0.7:
            maintenance.append("improve_routes — entries are poorly connected")
        if weakest < median_a // 4:
            maintenance.append("decay_unstable — weak entries have noise tokens")
        if len(self.entries) > 8:
            maintenance.append("suggest_fusion — check for near-duplicates")
            maintenance.append("suggest_fission — check for overloaded entries")

        action = (
            "grow — fill voids"          if h["score"] > 50 else
            "seek — expand sphere"       if h["score"] > 20 else
            "consolidate — run maintenance" if maintenance else
            "rest — corpus is healthy"
        )

        by_domain: dict[str, int] = {}
        for e in self.entries:
            d = e.get("domain", "?")
            by_domain[d] = by_domain.get(d, 0) + 1
        return {
            "entries":         len(self.entries),
            "domains":         len(by_domain),
            "self_cost":       round(sc, 4),
            "fill_rate":       len(self.entries) * 100 // IDEAL_CORPUS,
            "entropy":         round(self.sorter.corpus_entropy(), 3),
            "hunger":          h["score"],
            "routing_quality": quality,
            "action":          action,
            "maintenance":     maintenance,
        }

    def stability(self, k: int = 5) -> list[dict]:
        """Nuclear binding energy per entry. Sorted radioactive → iron."""
        if self.matrix is None or not self.entries:
            return []

        M         = self.matrix.astype(np.int64)
        self_amps = np.array([int(np.dot(M[i], M[i])) for i in range(len(M))])
        gram      = M @ M.T
        k         = max(1, min(k, len(self.entries) - 1))

        binding: list[int] = []
        for i in range(len(self.entries)):
            row    = gram[i].copy()
            row[i] = np.iinfo(np.int64).min
            top_k  = np.partition(row, -k)[-k:]
            binding.append(int(top_k.sum()) * MAX_EMBED_VAL // max(1, int(self_amps[i])))

        b_arr = np.array(binding, dtype=np.int64)
        n     = len(b_arr)
        s     = np.sort(b_arr)
        p10   = int(s[n * 10 // 100])
        p90   = int(s[n * 90 // 100])
        mid   = (p10 + p90) // 2

        def classify(b: int, conf: str) -> str:
            if conf == "ground-truth" or b >= p90: return "iron"
            if b >= mid:                            return "stable"
            if b >= p10:                            return "light"
            return "radioactive"

        result = [
            {"id": e.get("id",""), "domain": e.get("domain",""),
             "confidence": e.get("confidence","candidate"),
             "binding": int(b), "element": classify(int(b), e.get("confidence","candidate")),
             "claim": e.get("claim","")[:120]}
            for e, b in zip(self.entries, binding)
        ]
        result.sort(key=lambda x: x["binding"])
        return result

    def fuse(self, id_a: str, id_b: str) -> dict:
        """Fuse two light/stable entries. Returns synthesis suggestion."""
        ea = next((e for e in self.entries if e.get("id") == id_a), None)
        eb = next((e for e in self.entries if e.get("id") == id_b), None)
        if not ea: return {"error": f"not found: {id_a}"}
        if not eb: return {"error": f"not found: {id_b}"}

        stab   = {s["id"]: s["element"] for s in self.stability()}
        elem_a = stab.get(id_a, "stable")
        elem_b = stab.get(id_b, "stable")
        if "iron"        in (elem_a, elem_b): return {"error": "cannot fuse iron"}
        if "radioactive" in (elem_a, elem_b): return {"error": "heal radioactive entries first"}

        M   = self.matrix.astype(np.int64)
        ia  = next(i for i, e in enumerate(self.entries) if e.get("id") == id_a)
        ib  = next(i for i, e in enumerate(self.entries) if e.get("id") == id_b)
        mid = ((M[ia] + M[ib]) // 2).astype(np.int32)
        m   = int(np.max(np.abs(mid)))
        if m > MAX_EMBED_VAL:
            mid = (mid.astype(np.int64) * MAX_EMBED_VAL // m).astype(np.int32)

        amps         = M @ mid.astype(np.int64)
        amps[[ia,ib]] = 0
        bridge       = self.entries[int(np.argmax(amps))].get("claim","")[:100]

        return {
            "fusing":  [id_a, id_b],
            "claim_a": ea.get("claim","")[:120],
            "claim_b": eb.get("claim","")[:120],
            "bridge":  bridge,
            "instruction": (
                f"Write a new entry synthesizing '{id_a}' and '{id_b}'.\n"
                f"Nearest concept at midpoint: '{bridge}'\n"
                f"  python3 know.py add <domain> <subdomain> <id> \"<claim>\""
            )
        }

    def fission(self, entry_id: str, execute: bool = False) -> dict:
        """Split a heavy entry into two simpler ones.

        execute=False: returns fragments for review.
        execute=True: writes fragments, removes original, reloads.
        """
        e = next((x for x in self.entries if x.get("id") == entry_id), None)
        if not e:
            return {"error": f"not found: {entry_id}"}

        claim     = e.get("claim", "")
        sentences = [s.strip() for s in claim.replace(". ", ".|").replace(".\n", ".|").split("|")
                     if len(s.strip()) > 20]
        if len(sentences) < 2:
            return {"error": "entry too short to fission — only one sentence"}

        mid    = len(sentences) // 2
        part_a = ". ".join(sentences[:mid])
        part_b = ". ".join(sentences[mid:])
        id_a   = f"{entry_id}-a"
        id_b   = f"{entry_id}-b"
        domain = e.get("domain", "general")

        if execute:
            # Remove original (guarded json.loads)
            for f in sorted(self.dir.rglob("*.jsonl")):
                raw     = f.read_text()
                kept    = []
                removed = False
                for l in raw.splitlines():
                    s = l.strip()
                    if s.startswith("{"):
                        try:
                            if json.loads(s).get("id") == entry_id:
                                removed = True
                                continue
                        except json.JSONDecodeError:
                            pass
                    kept.append(l)
                if removed:
                    f.write_text("\n".join(kept) + "\n")
                    break
            # Write fragments
            path = self.dir / f"{domain}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            for fid, fc in ((id_a, part_a), (id_b, part_b)):
                with open(path, "a") as f:
                    f.write(json.dumps({"id": fid, "claim": fc, "domain": domain,
                                        "tags": e.get("tags",[]),
                                        "confidence": "verified"}) + "\n")
            self.reload()
            return {"executed": True, "removed": entry_id, "added": [id_a, id_b],
                    "fragment_a": part_a[:120], "fragment_b": part_b[:120]}

        stab    = {s["id"]: s for s in self.stability()}
        info    = stab.get(entry_id, {})
        return {
            "entry_id":   entry_id,
            "element":    info.get("element", "?"),
            "binding":    info.get("binding", 0),
            "original":   claim[:200],
            "fragment_a": part_a[:200],
            "fragment_b": part_b[:200],
            "instruction": (
                f"Call fission with execute=true to split automatically, or:\n"
                f"  know.py add {domain} <sub> {id_a} \"{part_a[:80]}\"\n"
                f"  know.py add {domain} <sub> {id_b} \"{part_b[:80]}\"\n"
                f"  know.py rm {entry_id}"
            )
        }

    def suggest_fission(self) -> list[dict]:
        """Find entries that are too heavy (bridge too many domains). Candidates for split.

        Detects entries whose tokens span multiple clusters — they're doing
        too much work. Splitting them into focused entries improves routing.
        """
        if not self.entries or self.matrix is None or len(self.entries) < 6:
            return []

        M = self.matrix.astype(np.int64)
        gram = M @ M.T
        np.fill_diagonal(gram, 0)

        candidates = []
        for i, e in enumerate(self.entries):
            # Heavy = high connectivity to many entries (bridges too much)
            self_amp = int(np.dot(M[i], M[i]))
            if self_amp == 0:
                continue
            # Count how many entries this one aligns with strongly (>30% self-amp)
            strong_connections = int((gram[i] > self_amp * 3 // 10).sum())
            # Also check: does the entry have many tokens? (long = complex)
            n_tokens = len(e.get("route", e.get("claim", "")).split())
            if strong_connections >= 4 and n_tokens >= 6:
                candidates.append({
                    "id": e.get("id", f"entry_{i}"),
                    "claim": e.get("claim", "")[:120],
                    "connections": strong_connections,
                    "tokens": n_tokens,
                    "reason": "bridges too many entries — consider splitting into focused parts",
                })

        candidates.sort(key=lambda x: -x["connections"])
        return candidates[:5]

    def suggest_fusion(self) -> list[dict]:
        """Find entry pairs that are near-duplicates. Candidates for merge.

        Detects pairs with very high mutual alignment (>80% of self-amplitude).
        Merging them reduces redundancy and frees geometric space.
        """
        if not self.entries or self.matrix is None or len(self.entries) < 4:
            return []

        M = self.matrix.astype(np.int64)
        gram = M @ M.T
        self_amps = np.array([int(np.dot(M[i], M[i])) for i in range(len(M))])

        candidates = []
        seen = set()
        for i in range(len(self.entries)):
            for j in range(i + 1, len(self.entries)):
                if (i, j) in seen:
                    continue
                # Mutual alignment as fraction of geometric mean of self-amps
                mutual = int(gram[i, j])
                threshold = int((self_amps[i] * self_amps[j]) ** 0.5 * 0.8)
                if mutual > threshold and mutual > 0:
                    seen.add((i, j))
                    candidates.append({
                        "id_a": self.entries[i].get("id", f"entry_{i}"),
                        "id_b": self.entries[j].get("id", f"entry_{j}"),
                        "claim_a": self.entries[i].get("claim", "")[:80],
                        "claim_b": self.entries[j].get("claim", "")[:80],
                        "similarity": round(mutual / max(1, threshold), 3),
                        "reason": "near-duplicates — consider merging into one stronger entry",
                    })

        candidates.sort(key=lambda x: -x["similarity"])
        return candidates[:5]

    def gaps(self, pair_idx: int = 0) -> dict:
        """Find geometric voids — isolated entry pairs with no knowledge between them.

        Janet finds the gap. The LLM writes the bridging entry.
        pair_idx: 0 = biggest void, 1 = second biggest, etc.
        """
        if self.matrix is None or len(self.entries) < 4:
            return {"error": "corpus too small"}

        M        = self.matrix.astype(np.int64)
        gram     = M @ M.T
        np.fill_diagonal(gram, np.iinfo(np.int64).max)
        order    = np.argsort(gram.min(axis=1))

        needed = pair_idx * 2 + 2
        if len(order) < needed:
            pair_idx = 0
            needed = 2
        i0 = int(order[0])
        i1 = int(order[min(1, len(order) - 1)])
        if pair_idx > 0 and len(order) >= pair_idx * 2 + 2:
            i0 = int(order[pair_idx * 2])
            i1 = int(order[pair_idx * 2 + 1])
        e0 = self.entries[i0]
        e1 = self.entries[i1]

        mid = ((M[i0] + M[i1]) // 2).astype(np.int32)
        m   = int(np.max(np.abs(mid)))
        if m > MAX_EMBED_VAL:
            mid = (mid.astype(np.int64) * MAX_EMBED_VAL // m).astype(np.int32)

        mid_amps = M @ mid.astype(np.int64)
        nearest  = self.entries[int(np.argmax(mid_amps))].get("claim","")[:120]

        return {
            "void_between":    [e0["id"], e1["id"]],
            "entry_a":         e0["claim"][:160],
            "entry_b":         e1["claim"][:160],
            "nearest_concept": nearest,
            "instruction": (
                f"Void between '{e0['id']}' ({e0.get('domain','')}) "
                f"and '{e1['id']}' ({e1.get('domain','')}).\n"
                f"Nearest concept at midpoint: '{nearest[:80]}'\n"
                f"  python3 know.py add <domain> <subdomain> <id> \"<claim>\""
            )
        }

    def seek(self) -> dict:
        """Look outward: what should the corpus learn next?

        Uses miss memory (failed queries) when available — real demand first.
        Falls back to geometric outward direction from corpus center.
        """
        if self.matrix is None or not self.entries:
            return {"error": "corpus too small"}

        M = self.matrix.astype(np.int64)

        if self._miss_vec is not None:
            direction = self._miss_vec.astype(np.int64)
            source    = "miss_memory"
        else:
            center    = M.sum(axis=0) // len(self.entries)
            direction = -center
            source    = "geometry"

        m = int(np.max(np.abs(direction)))
        if m > 0:
            direction = direction * MAX_EMBED_VAL // m

        edge_idx = int(np.argmax(M @ direction.astype(np.int64)))
        edge_e   = self.entries[edge_idx]

        dir_vec = direction.astype(np.int32)
        m       = int(np.max(np.abs(dir_vec)))
        if m > MAX_EMBED_VAL:
            dir_vec = (dir_vec.astype(np.int64) * MAX_EMBED_VAL // m).astype(np.int32)

        return {
            "source":      source,
            "edge_entry":  edge_e.get("id",""),
            "edge_claim":  edge_e.get("claim","")[:120],
            "direction":   _generate_from_edge(dir_vec),
            "instruction": (
                f"Seek direction ({source}): beyond '{edge_e.get('id','')}'\n"
                f"Add entries in this direction to grow the corpus."
            )
        }

    def stats(self) -> str:
        if not self.entries:
            return "(no entries)"
        by_domain: dict[str, int] = {}
        for e in self.entries:
            d = e.get("domain", "?")
            by_domain[d] = by_domain.get(d, 0) + 1
        lines = [f"{d}: {n}" for d, n in sorted(by_domain.items())]
        lines.insert(0, f"total: {len(self.entries)} entries, {len(by_domain)} domains")
        return "\n".join(lines)

    def total_entries(self) -> int:
        return len(self.entries)


# ── Session store ─────────────────────────────────────────────────────────

# ── Center registry ───────────────────────────────────────────────────────

_centers: dict[tuple[str, str], KnowledgeStore] = {}

def get_center(janet_dir: Path, center: str = "knowledge") -> KnowledgeStore:
    """Return (or create) the KnowledgeStore for a given center."""
    key = (str(janet_dir), center)
    if key not in _centers:
        _centers[key] = KnowledgeStore(janet_dir / center)
    return _centers[key]


def get_all_centers(janet_dir: Path) -> dict[str, KnowledgeStore]:
    """Load all centers in .janet/ directory. Each subdirectory with .jsonl files is a center."""
    if not janet_dir.exists():
        return {}
    for subdir in sorted(janet_dir.iterdir()):
        if subdir.is_dir() and any(subdir.rglob("*.jsonl")):
            get_center(janet_dir, subdir.name)
    return {k[1]: v for k, v in _centers.items() if k[0] == str(janet_dir)}


def route_query(janet_dir: Path, text: str) -> tuple[str, KnowledgeStore]:
    """GWT routing: find the center whose corpus is most aligned with the query.

    Each center is a node in the network. The query goes to the specialist
    with highest amplitude (lowest cost). If no center has good alignment,
    returns the default 'knowledge' center.

    This is automatic — the LLM doesn't need to specify center.
    """
    centers = get_all_centers(janet_dir)
    if not centers:
        return "knowledge", get_center(janet_dir, "knowledge")

    best_center = "knowledge"
    best_amp = -(10**18)

    for name, store in centers.items():
        if not store.entries or store.matrix is None:
            continue
        qvec = store.sorter.encode(text)
        amps = store.matrix.astype(np.int64) @ qvec.astype(np.int64)
        max_amp = int(amps.max())
        if max_amp > best_amp:
            best_amp = max_amp
            best_center = name

    return best_center, get_center(janet_dir, best_center)


# ── Execution tools ───────────────────────────────────────────────────────

def exec_bash(command: str, cwd: str) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True,
                           text=True, timeout=30, cwd=cwd)
        out = (r.stdout + r.stderr).strip()
        return out[:4096] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "(timeout)"


def exec_read(path: str, cwd: str) -> str:
    full = path if os.path.isabs(path) else os.path.join(cwd, path)
    try:
        return open(full).read()[:8192]
    except Exception as e:
        return f"(error: {e})"


def exec_grep(pattern: str, path: str, cwd: str) -> str:
    target = path if os.path.isabs(path) else os.path.join(cwd, path)
    try:
        r = subprocess.run(["grep", "-rn", pattern, target],
                           capture_output=True, text=True, timeout=15)
        out = r.stdout.strip()
        return out[:4096] if out else "(no matches)"
    except Exception as e:
        return f"(error: {e})"


def exec_write(path: str, content: str, cwd: str, janet_dir: Path) -> str:
    full = path if os.path.isabs(path) else os.path.join(cwd, path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    if path.endswith(".jsonl"):
        store = get_center(janet_dir)
        store.reload()
        return f"wrote {path} (+corpus reloaded, {store.total_entries()} entries)"
    return f"wrote {path}"


def exec_edit(path: str, old_str: str, new_str: str, cwd: str, janet_dir: Path) -> str:
    full = path if os.path.isabs(path) else os.path.join(cwd, path)
    try:
        original = open(full).read()
    except Exception as e:
        return f"(error reading {path}: {e})"
    if old_str not in original:
        return f"(old_string not found in {path})"
    updated = original.replace(old_str, new_str, 1)
    with open(full, "w") as f:
        f.write(updated)
    if path.endswith(".jsonl"):
        store = get_center(janet_dir)
        store.reload()
        return f"edited {path} (+corpus reloaded, {store.total_entries()} entries)"
    return f"edited {path}"


# ── MCP tool definitions ──────────────────────────────────────────────────

TOOLS = [
    {
        "name": "query",
        "description": (
            "Query the knowledge network. Returns the full orbit.\n"
            "Auto-routes to the best center (GWT: geometry picks the specialist).\n"
            "center='auto' (default) — automatic routing across all centers.\n"
            "center='knowledge' — force specific center.\n"
            "center='{name}' — any custom .janet/{name}/ corpus.\n"
            "3–7 distinctive words. Avoid stop words."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "q":      {"type": "string"},
                            "center": {"type": "string", "default": "auto"},
                        },
                        "required": ["q"]}
    },
    {
        "name": "add",
        "description": (
            "Add an entry to a knowledge center (.janet/{center}/{domain}.jsonl).\n"
            "center='knowledge' (default). Immune system rejects foreign entries "
            "unless confidence='verified' or 'ground-truth'.\n"
            "Use 'route' for explicit routing tags (flat keywords for geometry).\n"
            "If no route, call suggest_route first to get geometry-optimal tags."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "domain":     {"type": "string"},
                            "id":         {"type": "string"},
                            "claim":      {"type": "string"},
                            "route":      {"type": "string",
                                          "description": "Flat routing tags for geometry. Use suggest_route to generate."},
                            "tags":       {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "string",
                                          "description": "verified | ground-truth | candidate"},
                            "source":     {"type": "string"},
                            "center":     {"type": "string", "default": "knowledge"},
                        },
                        "required": ["domain", "id", "claim", "route"]}
    },
    {
        "name": "suggest_route",
        "description": (
            "Suggest routing tags for a new entry based on corpus geometry.\n"
            "Returns flat keywords that will create optimal bridges to existing knowledge.\n"
            "Call before 'add' to get the best route for a new claim."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "claim":  {"type": "string"},
                            "center": {"type": "string", "default": "knowledge"},
                        },
                        "required": ["claim"]}
    },
    {
        "name": "improve_routes",
        "description": (
            "Find entries with weak geometric connections and suggest better routes.\n"
            "Returns entries whose routing tokens don't bridge well to the corpus.\n"
            "Use to maintain A4 (faithful encoding) — the system tells you what to fix."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "center": {"type": "string", "default": "knowledge"},
                        },
                        "required": []}
    },
    {
        "name": "gaps",
        "description": (
            "Find geometric voids — the two most isolated entries "
            "with no knowledge between them. Janet finds the gap; the LLM writes the entry."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "pair_idx": {"type": "integer",
                                        "description": "0=biggest void, 1=second, etc."}
                        }}
    },
    {
        "name": "seek",
        "description": (
            "Look outward: what should the corpus learn next?\n"
            "Uses miss memory (failed queries) when available, else geometric edge."
        ),
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "stability",
        "description": (
            "Nuclear binding energy per entry: iron / stable / light / radioactive.\n"
            "Iron = ground truth (never decays). Radioactive = heal candidates."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "k": {"type": "integer",
                                  "description": "Nearest neighbors (default 5)"}
                        }}
    },
    {
        "name": "fuse",
        "description": (
            "Fuse two light/stable entries into a new synthesis.\n"
            "Returns claim suggestion for review — does not auto-add.\n"
            "Cannot fuse iron (ground truth) or radioactive entries."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "id_a": {"type": "string"},
                            "id_b": {"type": "string"}
                        },
                        "required": ["id_a", "id_b"]}
    },
    {
        "name": "fission",
        "description": (
            "Split a heavy entry into two simpler ones.\n"
            "execute=false (default): returns fragments for review.\n"
            "execute=true: writes both fragments, removes original."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "id":      {"type": "string"},
                            "execute": {"type": "boolean", "description": "Default false"}
                        },
                        "required": ["id"]}
    },
    {
        "name": "suggest_fission",
        "description": (
            "Find entries that bridge too many domains (too heavy).\n"
            "Geometry detects candidates for splitting. Returns list with reasons."
        ),
        "inputSchema": {"type": "object",
                        "properties": {"center": {"type": "string", "default": "knowledge"}},
                        "required": []}
    },
    {
        "name": "suggest_fusion",
        "description": (
            "Find near-duplicate entry pairs (too similar).\n"
            "Geometry detects candidates for merging. Returns pairs with similarity score."
        ),
        "inputSchema": {"type": "object",
                        "properties": {"center": {"type": "string", "default": "knowledge"}},
                        "required": []}
    },
    {
        "name": "hunger",
        "description": "Corpus need signal. Score 0–100: >50=grow, >20=seek, else rest.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "health",
        "description": "Full self-assessment: entries, fill rate, hunger, routing quality, action.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "discover",
        "description": (
            "Generate novel connections from corpus geometry.\n"
            "Input: a query or topic. Output: corpus fragments that together\n"
            "point toward something implicit but never stated.\n"
            "Use when: you want to find what the corpus IMPLIES but doesn't say."
        ),
        "inputSchema": {"type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"]}
    },
    {
        "name": "self_cost",
        "description": (
            "Measure corpus integration (internal resonance).\n"
            "self_cost < 0.5 = integrated. > 0.7 = fragmented.\n"
            "Stronger predictor of accuracy than phi (r=0.82). O(N²) — works for large corpora."
        ),
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "vocabulary",
        "description": (
            "Return Janet's most distinctive tokens — the routing basis.\n"
            "Use these tokens in queries for optimal retrieval.\n"
            "Distinctive = appears in 2-15 entries (not noise, not ubiquitous)."
        ),
        "inputSchema": {"type": "object",
                        "properties": {
                            "top_n": {"type": "integer", "description": "How many tokens (default 50)"}
                        }}
    },
    {
        "name": "reset_session",
        "description": "Clear session state: anti-gravity (prevents repetition) and miss memory.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "stats",
        "description": "Corpus entries per domain.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "reload",
        "description": "Reload corpus from .jsonl files.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "write",
        "description": "Write a file. .jsonl files trigger corpus reload.",
        "inputSchema": {"type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"]}
    },
    {
        "name": "edit",
        "description": "Replace exact string in a file. .jsonl files trigger corpus reload.",
        "inputSchema": {"type": "object",
                        "properties": {
                            "path":       {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"}
                        },
                        "required": ["path", "old_string", "new_string"]}
    },
    {
        "name": "bash",
        "description": "Execute a shell command.",
        "inputSchema": {"type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]}
    },
    {
        "name": "read",
        "description": "Read raw file bytes. Last resort — prefer query.",
        "inputSchema": {"type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]}
    },
    {
        "name": "grep",
        "description": "Exact regex search. Last resort — prefer query.",
        "inputSchema": {"type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path":    {"type": "string"}
                        },
                        "required": ["pattern"]}
    },
]


# ── MCP protocol (stdio) ──────────────────────────────────────────────────

def handle(req: dict, cwd: str, janet_dir: Path) -> dict | None:
    method = req.get("method", "")
    rid    = req.get("id")
    params = req.get("params", {})
    store  = get_center(janet_dir)

    if method == "initialize":
        sys.stderr.write(f"[janet] {store.stats().splitlines()[0]}\n")
        sys.stderr.flush()
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {"tools": {"listChanged": False}},
                "serverInfo":      {"name": "janet", "version": "6.0"}
            }
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})

        center = args.get("center", "auto")
        if center == "auto" and name == "query":
            # GWT routing: automatically find the best center for this query
            routed_center, store = route_query(janet_dir, args.get("q", ""))
        else:
            if center == "auto":
                center = "knowledge"
            store = get_center(janet_dir, center)
            routed_center = center

        if   name == "query":
            q = args.get("q", "")
            answer = store.query(q) or "(outside sphere)"
            # Add routing info if auto-routed
            if args.get("center", "auto") == "auto" and routed_center != "knowledge":
                answer = f"[routed→{routed_center}] {answer}"
            result = answer
        elif name == "add":
            entry = {k: v for k, v in {
                "id": args.get("id",""), "claim": args.get("claim",""),
                "route": args.get("route",""),
                "domain": args.get("domain",""), "tags": args.get("tags",[]),
                "confidence": args.get("confidence",""), "source": args.get("source",""),
            }.items() if v}
            result = get_center(janet_dir, center).add(args.get("domain", "general"), entry)
        elif name == "suggest_route":
            result = store.suggest_route(args.get("claim", ""))
        elif name == "improve_routes":
            result = json.dumps(store.improve_routes(), indent=2, ensure_ascii=False)
        elif name == "gaps":          result = json.dumps(store.gaps(args.get("pair_idx",0)), indent=2, ensure_ascii=False)
        elif name == "seek":          result = json.dumps(store.seek(), indent=2, ensure_ascii=False)
        elif name == "stability":     result = json.dumps(store.stability(args.get("k",5)), indent=2, ensure_ascii=False)
        elif name == "fuse":          result = json.dumps(store.fuse(args.get("id_a",""), args.get("id_b","")), indent=2, ensure_ascii=False)
        elif name == "fission":       result = json.dumps(store.fission(args.get("id",""), args.get("execute",False)), indent=2, ensure_ascii=False)
        elif name == "suggest_fission": result = json.dumps(store.suggest_fission(), indent=2, ensure_ascii=False)
        elif name == "suggest_fusion":  result = json.dumps(store.suggest_fusion(), indent=2, ensure_ascii=False)
        elif name == "hunger":        result = json.dumps(store.hunger(), indent=2)
        elif name == "health":        result = json.dumps(store.health(), indent=2)
        elif name == "discover":
            q = args.get("q", "")
            state = store.sorter.encode(q)
            generated = store.sorter.generate_coherent(state)
            if generated:
                result = f"[discovery] {generated}"
            else:
                result = "(no novel direction found — query may be outside corpus or too aligned with centroid)"
        elif name == "self_cost":
            sc_val = store.sorter.self_cost()
            result = json.dumps({"self_cost": round(sc_val, 4),
                                 "interpretation": (
                                     "integrated" if sc_val < 0.5 else
                                     "fragmented" if sc_val > 0.7 else
                                     "moderate"
                                 )}, indent=2)
        elif name == "vocabulary":
            top_n = args.get("top_n", 50)
            vocab = store.sorter.vocabulary(top_n)
            result = " ".join(vocab) if vocab else "(no distinctive tokens yet)"
        elif name == "reset_session": result = store.reset_session()
        elif name == "stats":         result = store.stats()
        elif name == "reload":        store.reload(); result = store.stats()
        elif name == "write":         result = exec_write(args.get("path",""), args.get("content",""), cwd, janet_dir)
        elif name == "edit":          result = exec_edit(args.get("path",""), args.get("old_string",""), args.get("new_string",""), cwd, janet_dir)
        elif name == "bash":          result = exec_bash(args.get("command",""), cwd)
        elif name == "read":          result = exec_read(args.get("path",""), cwd)
        elif name == "grep":          result = exec_grep(args.get("pattern",""), args.get("path","."), cwd)
        else:                         result = f"unknown tool: {name}"

        return {"jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": result}]}}

    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"unknown method: {method}"}}


def main() -> None:
    cwd        = os.environ.get("JANET_DIR") or os.getcwd()
    janet_dir  = resolve_janet(cwd)
    sys.stderr.write(f"[janet-mcp] cwd={cwd}  .janet={janet_dir}\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req  = json.loads(line)
            resp = handle(req, cwd, janet_dir)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"[janet-mcp] error: {e}\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()
