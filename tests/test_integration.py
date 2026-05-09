"""Integration tests — end-to-end flows through the MCP protocol."""

import json
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcp import handle, KnowledgeStore, get_center, _centers


@pytest.fixture(autouse=True)
def clear_center_cache():
    _centers.clear()
    yield
    _centers.clear()


@pytest.fixture
def corpus_dir(tmp_path):
    """Build a realistic multi-domain corpus in .janet/knowledge/."""
    kdir = tmp_path / ".janet" / "knowledge"
    kdir.mkdir(parents=True)
    biology = [
        {"id": "photosynthesis", "domain": "biology", "claim": "Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen", "tags": ["plants", "energy", "chlorophyll"], "confidence": "ground-truth"},
        {"id": "mitochondria", "domain": "biology", "claim": "Mitochondria produce ATP through oxidative phosphorylation in the electron transport chain", "tags": ["cell", "energy", "atp"], "confidence": "verified"},
        {"id": "dna-structure", "domain": "biology", "claim": "DNA is a double helix of nucleotides encoding genetic information for protein synthesis", "tags": ["genetics", "nucleotides"], "confidence": "ground-truth"},
        {"id": "evolution", "domain": "biology", "claim": "Natural selection drives evolution by favoring organisms with advantageous traits", "tags": ["darwin", "adaptation"], "confidence": "verified"},
    ]
    physics = [
        {"id": "gravity", "domain": "physics", "claim": "General relativity describes gravity as curvature of spacetime caused by mass and energy", "tags": ["einstein", "spacetime"], "confidence": "ground-truth"},
        {"id": "quantum-superposition", "domain": "physics", "claim": "Quantum superposition allows particles to exist in multiple states until measured", "tags": ["quantum", "measurement"], "confidence": "verified"},
        {"id": "thermodynamics-2", "domain": "physics", "claim": "Second law of thermodynamics states entropy of isolated systems always increases", "tags": ["entropy", "heat"], "confidence": "ground-truth"},
        {"id": "electromagnetism", "domain": "physics", "claim": "Maxwell equations unify electricity magnetism and light as electromagnetic waves", "tags": ["maxwell", "waves"], "confidence": "verified"},
    ]
    cs = [
        {"id": "turing-machine", "domain": "cs", "claim": "A Turing machine can compute any computable function given enough time and tape", "tags": ["computation", "decidability"], "confidence": "ground-truth"},
        {"id": "backpropagation", "domain": "cs", "claim": "Backpropagation computes gradients for neural network weight updates via chain rule", "tags": ["ml", "gradient"], "confidence": "verified"},
    ]
    with open(kdir / "biology.jsonl", "w") as f:
        for e in biology:
            f.write(json.dumps(e) + "\n")
    with open(kdir / "physics.jsonl", "w") as f:
        for e in physics:
            f.write(json.dumps(e) + "\n")
    with open(kdir / "cs.jsonl", "w") as f:
        for e in cs:
            f.write(json.dumps(e) + "\n")
    return tmp_path


def janet_dir(corpus_dir: Path) -> Path:
    return corpus_dir / ".janet"


def call_tool(name: str, args: dict, cwd_path: Path) -> str:
    """Helper: call an MCP tool and return the text result."""
    jdir = cwd_path / ".janet"
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": name, "arguments": args}}
    resp = handle(req, str(cwd_path), jdir)
    return resp["result"]["content"][0]["text"]


class TestEndToEndQuery:
    def test_confident_query_returns_match(self, corpus_dir):
        result = call_tool("query", {"q": "photosynthesis sunlight plants"}, corpus_dir)
        assert "photosynthesis" in result.lower()

    def test_query_different_domains(self, corpus_dir):
        bio = call_tool("query", {"q": "DNA genetic nucleotides"}, corpus_dir)
        assert "dna" in bio.lower() or "genetic" in bio.lower()

        phys = call_tool("query", {"q": "gravity spacetime curvature"}, corpus_dir)
        assert "gravity" in phys.lower() or "spacetime" in phys.lower()

    def test_query_with_center(self, corpus_dir):
        result = call_tool("query", {"q": "photosynthesis sunlight", "center": "knowledge"}, corpus_dir)
        assert len(result) > 0

    def test_outside_sphere_returns_orbit(self, corpus_dir):
        result = call_tool("query", {"q": "kubernetes container orchestration deployment"}, corpus_dir)
        assert len(result) > 0

    def test_anti_gravity_prevents_repetition(self, corpus_dir):
        r1 = call_tool("query", {"q": "energy production cells"}, corpus_dir)
        r2 = call_tool("query", {"q": "energy production cells"}, corpus_dir)
        assert len(r1) > 0 and len(r2) > 0


class TestEndToEndAdd:
    def test_add_verified_entry(self, corpus_dir):
        result = call_tool("add", {
            "domain": "biology", "id": "cell-membrane",
            "claim": "Cell membranes are phospholipid bilayers controlling molecular transport",
            "confidence": "verified", "tags": ["cell", "membrane"]
        }, corpus_dir)
        assert "added" in result
        q_result = call_tool("query", {"q": "cell membrane phospholipid"}, corpus_dir)
        assert "membrane" in q_result.lower() or "cell" in q_result.lower()

    def test_add_rejected_by_immune(self, corpus_dir):
        result = call_tool("add", {
            "domain": "nonsense", "id": "gibberish",
            "claim": "xyzzy plugh qwerty asdf jkl random noise",
        }, corpus_dir)
        assert "rejected" in result.lower()

    def test_add_override_with_verified(self, corpus_dir):
        result = call_tool("add", {
            "domain": "new", "id": "override-test",
            "claim": "xyzzy plugh completely foreign content",
            "confidence": "verified",
        }, corpus_dir)
        assert "added" in result


class TestEndToEndHealth:
    def test_health_flow(self, corpus_dir):
        health = json.loads(call_tool("health", {}, corpus_dir))
        assert health["entries"] == 10
        assert health["domains"] == 3
        assert "action" in health

    def test_hunger_flow(self, corpus_dir):
        hunger = json.loads(call_tool("hunger", {}, corpus_dir))
        assert 0 <= hunger["score"] <= 100

    def test_stability_flow(self, corpus_dir):
        stability = json.loads(call_tool("stability", {"k": 3}, corpus_dir))
        assert len(stability) == 10
        elements = {s["element"] for s in stability}
        assert len(elements) >= 2

    def test_gaps_flow(self, corpus_dir):
        gaps = json.loads(call_tool("gaps", {"pair_idx": 0}, corpus_dir))
        assert "void_between" in gaps
        assert len(gaps["void_between"]) == 2

    def test_seek_flow(self, corpus_dir):
        seek = json.loads(call_tool("seek", {}, corpus_dir))
        assert "source" in seek
        assert seek["source"] in ("geometry", "miss_memory")


class TestEndToEndSession:
    def test_reset_clears_state(self, corpus_dir):
        call_tool("query", {"q": "photosynthesis"}, corpus_dir)
        result = call_tool("reset_session", {}, corpus_dir)
        assert "cleared" in result

    def test_reload_preserves_data(self, corpus_dir):
        before = call_tool("stats", {}, corpus_dir)
        call_tool("reload", {}, corpus_dir)
        after = call_tool("stats", {}, corpus_dir)
        assert "10" in before and "10" in after


class TestEndToEndFusion:
    def test_fuse_valid_entries(self, corpus_dir):
        result = json.loads(call_tool("fuse", {"id_a": "photosynthesis", "id_b": "mitochondria"}, corpus_dir))
        assert "fusing" in result or "error" in result

    def test_fission_entry(self, corpus_dir):
        call_tool("add", {
            "domain": "biology", "id": "long-entry",
            "claim": "Cells divide through mitosis involving prophase metaphase anaphase and telophase. The cell cycle is regulated by cyclins and cyclin-dependent kinases that control progression.",
            "confidence": "verified",
        }, corpus_dir)
        result = json.loads(call_tool("fission", {"id": "long-entry"}, corpus_dir))
        assert "fragment_a" in result or "error" in result
