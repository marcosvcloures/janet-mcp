"""Unit tests for mcp.py — KnowledgeStore and MCP protocol."""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp import (
    entry_text, KnowledgeStore, handle, get_center, _centers,
    exec_bash, exec_read, exec_write, exec_edit,
    TOOLS,
)
from janet import DIMS, MAX_EMBED_VAL


# ── entry_text ────────────────────────────────────────────────────────────

class TestEntryText:
    def test_formats_entry(self):
        e = {"domain": "physics", "subdomain": "quantum", "id": "qm-1",
             "tags": ["wave", "particle"], "claim": "Light is both wave and particle"}
        t = entry_text(e)
        assert "physics" in t
        assert "quantum" in t
        assert "qm-1" in t
        assert "Light is both wave and particle" in t

    def test_handles_missing_fields(self):
        e = {"claim": "something"}
        t = entry_text(e)
        assert "something" in t


# ── KnowledgeStore ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_janet(tmp_path):
    """Create a temp .janet/knowledge/ dir with sample entries."""
    kdir = tmp_path / ".janet" / "knowledge"
    kdir.mkdir(parents=True)
    entries = [
        {"id": "photosynthesis", "domain": "biology", "claim": "Plants convert sunlight to energy via photosynthesis", "tags": ["plants", "energy"], "confidence": "verified"},
        {"id": "gravity", "domain": "physics", "claim": "Gravity is the curvature of spacetime caused by mass", "tags": ["gravity", "spacetime"], "confidence": "verified"},
        {"id": "dna-replication", "domain": "biology", "claim": "DNA replicates through semiconservative mechanism using polymerase", "tags": ["dna", "genetics"], "confidence": "verified"},
        {"id": "entropy", "domain": "physics", "claim": "Entropy measures disorder and always increases in closed systems", "tags": ["thermodynamics"], "confidence": "verified"},
        {"id": "neural-networks", "domain": "cs", "claim": "Neural networks learn by adjusting weights through backpropagation", "tags": ["ml", "ai"], "confidence": "verified"},
    ]
    with open(kdir / "test.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return tmp_path / ".janet"


@pytest.fixture(autouse=True)
def clear_center_cache():
    _centers.clear()
    yield
    _centers.clear()


@pytest.fixture
def store(tmp_janet):
    return KnowledgeStore(tmp_janet / "knowledge")


class TestKnowledgeStore:
    def test_reload_loads_entries(self, store):
        assert len(store.entries) == 5
        assert store.matrix is not None
        assert store.matrix.shape[0] == 5
        assert store.matrix.shape[1] == store.sorter.dims

    def test_query_returns_relevant(self, store):
        result = store.query("plants sunlight energy")
        assert "photosynthesis" in result.lower()

    def test_query_empty_corpus(self, tmp_path):
        kdir = tmp_path / "empty_knowledge"
        kdir.mkdir()
        s = KnowledgeStore(kdir)
        assert s.query("anything") == ""

    def test_reject_foreign(self, store):
        assert store.reject("xyzzy plugh qwerty asdf") is True

    def test_reject_coherent(self, store):
        assert store.reject("photosynthesis energy sunlight plants") is False

    def test_hunger_returns_dict(self, store):
        h = store.hunger()
        assert "score" in h
        assert 0 <= h["score"] <= 100
        assert "message" in h

    def test_health_returns_dict(self, store):
        h = store.health()
        assert h["entries"] == 5
        assert "action" in h
        assert "hunger" in h

    def test_add_entry(self, store, tmp_janet):
        entry = {"id": "test-new", "claim": "Test claim about biology and cells",
                 "domain": "biology", "confidence": "verified"}
        result = store.add("biology", entry)
        assert "added" in result
        assert len(store.entries) == 6

    def test_add_rejected_foreign(self, store):
        entry = {"id": "foreign", "claim": "xyzzy plugh completely random gibberish words",
                 "domain": "nonsense"}
        result = store.add("nonsense", entry)
        assert "rejected" in result.lower()

    def test_reset_session(self, store):
        store._anti_grav = np.ones(DIMS, dtype=np.int32)
        store._miss_vec = np.ones(DIMS, dtype=np.int32)
        store.reset_session()
        assert store._anti_grav is None
        assert store._miss_vec is None

    def test_stats(self, store):
        s = store.stats()
        assert "5" in s
        assert "biology" in s

    def test_stability(self, store):
        result = store.stability(k=2)
        assert len(result) == 5
        for r in result:
            assert r["element"] in ("iron", "stable", "light", "radioactive")

    def test_gaps(self, store):
        result = store.gaps(pair_idx=0)
        assert "void_between" in result
        assert "instruction" in result

    def test_seek(self, store):
        result = store.seek()
        assert "source" in result
        assert "edge_entry" in result

    def test_fuse_not_found(self, store):
        result = store.fuse("nonexistent-a", "nonexistent-b")
        assert "error" in result

    def test_fission_not_found(self, store):
        result = store.fission("nonexistent")
        assert "error" in result


# ── MCP protocol handler ──────────────────────────────────────────────────

class TestMCPHandler:
    def test_initialize(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        assert resp["id"] == 1
        assert resp["result"]["serverInfo"]["name"] == "janet"

    def test_tools_list(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "query" in names
        assert "add" in names
        assert "health" in names

    def test_tools_call_query(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
               "params": {"name": "query", "arguments": {"q": "photosynthesis sunlight"}}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        text = resp["result"]["content"][0]["text"]
        assert len(text) > 0

    def test_tools_call_query_center(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
               "params": {"name": "query", "arguments": {"q": "gravity spacetime", "center": "knowledge"}}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        text = resp["result"]["content"][0]["text"]
        assert len(text) > 0

    def test_tools_call_health(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
               "params": {"name": "health", "arguments": {}}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        data = json.loads(resp["result"]["content"][0]["text"])
        assert data["entries"] == 5

    def test_tools_call_hunger(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
               "params": {"name": "hunger", "arguments": {}}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "score" in data

    def test_tools_call_stats(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
               "params": {"name": "stats", "arguments": {}}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        assert "5" in resp["result"]["content"][0]["text"]

    def test_unknown_method(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 7, "method": "nonexistent", "params": {}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        assert "error" in resp

    def test_unknown_tool(self, tmp_janet):
        req = {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
               "params": {"name": "nonexistent_tool", "arguments": {}}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        assert "unknown tool" in resp["result"]["content"][0]["text"]

    def test_notifications_initialized(self, tmp_janet):
        req = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        resp = handle(req, str(tmp_janet.parent), tmp_janet)
        assert resp is None


# ── Exec helpers ──────────────────────────────────────────────────────────

class TestExecHelpers:
    def test_exec_bash(self):
        result = exec_bash("echo hello", "/tmp")
        assert "hello" in result

    def test_exec_read(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content here")
        result = exec_read(str(f), str(tmp_path))
        assert "content here" in result

    def test_exec_read_missing(self):
        result = exec_read("/nonexistent/file.txt", "/tmp")
        assert "error" in result.lower()

    def test_exec_write(self, tmp_path):
        janet_dir = tmp_path / ".janet"
        janet_dir.mkdir()
        result = exec_write("out.txt", "written", str(tmp_path), janet_dir)
        assert "wrote" in result
        assert (tmp_path / "out.txt").read_text() == "written"

    def test_exec_edit(self, tmp_path):
        janet_dir = tmp_path / ".janet"
        janet_dir.mkdir()
        f = tmp_path / "edit.txt"
        f.write_text("old content here")
        result = exec_edit(str(f), "old", "new", str(tmp_path), janet_dir)
        assert "edited" in result
        assert "new content here" in f.read_text()
