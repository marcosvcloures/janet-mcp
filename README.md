# janet-mcp

Long-term memory for AI agents. Not search — memory.

RAG searches documents. Janet *remembers* verified knowledge, knows when it doesn't know, detects its own degradation, and tells you how to improve it. Use it alongside your RAG, not instead of it.

For the theoretical foundation and experiments, see [janet-py](https://github.com/marcosvcloures/janet-py).

---

## Why this exists

An LLM with RAG can search. An LLM with janet can *remember*.

| | RAG (search) | Janet (memory) |
|---|---|---|
| Purpose | Find relevant documents | Remember verified knowledge |
| Persistence | Stateless (re-embeds each time) | Persistent (survives sessions) |
| Confidence | None (returns top-k blindly) | Orbit cost + amplitude |
| "I don't know" | Never | amplitude=0 (native) |
| Health monitoring | None | self_cost detects degradation |
| Self-improvement | None | suggest_route, improve_routes |
| Grows with use | No (static index) | Yes (Hebbian + consolidation) |
| Determinism | No (float embeddings) | Yes (int32, reproducible) |

RAG is perception (searching the external world). Janet is hippocampus (internal verified memory). The LLM is cortex (reasoning over both). Together: an agent that perceives, remembers, and reasons.

Any LLM (Claude, GPT, local models) gets these capabilities via MCP without fine-tuning.

---

## Use cases

Janet is a knowledge persistence tool — not just for software engineering:

| Domain | What the corpus holds |
|---|---|
| Software projects | Architecture decisions, API contracts, known bugs |
| Research | Verified findings, dead ends, experimental parameters |
| Legal | Precedents, rulings, case-specific knowledge |
| Medical | Protocols, drug interactions, clinical decisions |
| Consulting | Client knowledge, project decisions, lessons learned |
| Personal | Anything verified that you don't want to re-derive |

The common pattern: **curated knowledge that grows over months, needs consistency, and benefits from "I don't know" + self-maintenance.**

---

## Janet + RAG (complementary, not competitive)

They are different layers of cognition:

```
External world (documents, web, APIs)
  → RAG searches, finds relevant text (perception)
    → LLM extracts facts, reasons about them (cortex)
      → Janet stores verified conclusions (long-term memory)
        → Next session: Janet remembers, RAG searches again
```

| Layer | Tool | Analogy |
|---|---|---|
| Perception | RAG / web search | Eyes, ears |
| Reasoning | LLM | Cortex |
| Long-term memory | Janet | Hippocampus |
| Executive function | Human operator | Prefrontal cortex |

**Use RAG for:** "find me documents about X" (broad, semantic, stateless).

**Use janet for:** "what have we established about X?" (precise, verified, persistent).

**Use both:** RAG finds → LLM reasons → operator verifies → janet remembers. Next session, janet provides context before RAG even searches.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  .janet/                                                    │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   auth   │  │   api    │  │  domain  │  │  infra   │   │
│  │  N=25    │  │  N=30    │  │  N=40    │  │  N=20    │   │
│  │  phi<1ms │  │  phi<1ms │  │  phi<1ms │  │  phi<1ms │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │              │         │
│       └──────────────┴──────┬───────┴──────────────┘         │
│                             │                               │
│                    ┌────────┴────────┐                       │
│                    │  GWT routing    │                       │
│                    │  (auto: query → │                       │
│                    │   best center)  │                       │
│                    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

Each center is a specialist with its own corpus, phi, and self_cost. Queries auto-route to the center with highest geometric alignment — no need to specify where to look.

---

## The symbiosis

```
Operator (executive function)
  — verifies, decides, judges quality
  — sovereign over what enters memory

LLM (cortex)
  — reasons, generates, translates
  — interprets orbits (cost → confidence, amplitude → domain)
  — executes prescriptions (add, fuse, fission, improve_routes)
  — runs consolidation when health() says so

Janet (hippocampus)
  — remembers across sessions (persistent, deterministic)
  — auto-routes queries to the right specialist center
  — returns full orbit (path, cost, amplitude, convergence)
  — diagnoses: self_cost, health, suggest_fission/fusion
  — prescribes: suggest_route, improve_routes
  — maintains: decay_unstable, immune rejection
  — CANNOT consolidate alone — depends on LLM

RAG (perception — optional, external)
  — searches documents, web, APIs
  — finds relevant text for the LLM to reason about
  — stateless — does not remember between sessions
```

The LLM reasons but forgets. Janet remembers but doesn't reason. The operator judges but can't process 500 entries. Together: an agent that perceives, remembers, reasons, and knows its own state.

```
Operator asks a question
  → LLM queries janet first (do we already know this?)
    → Janet returns orbit: answer + confidence
      → If amplitude > 0: "We established X" (memory)
      → If amplitude = 0: "We don't know yet" (honest gap)
  → If gap: LLM uses RAG / web / tools to find answer (perception)
    → LLM reasons about what it found (cortex)
      → Operator verifies
        → LLM calls suggest_route → add (memory formation)
          → Janet integrates, health() monitors
```

---

## Setup

```bash
pip install numpy
```

```json
{
  "mcpServers": {
    "janet": {
      "command": "python3",
      "args": ["/path/to/janet-mcp/mcp.py"],
      "env": { "JANET_DIR": "/path/to/your/project" }
    }
  }
}
```

---

## Knowledge structure

```
your-project/
  .janet/
    architecture/   ← system design decisions
    api/            ← endpoints, contracts, protocols
    domain/         ← business logic, rules
    infrastructure/ ← deployment, scaling, monitoring
    tools/          ← tool routing for the LLM
```

Each subdirectory is a **center** (specialist node). Queries auto-route across all centers.

Each `.jsonl` file contains one entry per line:
```json
{"id": "auth-1", "domain": "auth", "route": "authentication jwt stateless token session", "claim": "Authentication uses JWT. Stateless. No server-side sessions.", "confidence": "verified"}
```

**Dual representation:**
- `route`: flat tags for geometry (what janet sees). Subdomain tokens first.
- `claim`: natural language (what the LLM reads back).
- `domain`: which .jsonl file. Not part of the geometry.

---

## MCP Tools

### Retrieval
| Tool | Purpose |
|------|---------|
| `query` | Auto-routes to best center. Returns orbit: answer + cost + amplitude + convergence |
| `vocabulary` | Janet's distinctive tokens for optimal routing |

### Self-prescription
| Tool | Purpose |
|------|---------|
| `suggest_route` | Geometry recommends routing tokens for a new entry |
| `improve_routes` | Find weak entries + ubiquitous tokens to fix |
| `suggest_fission` | Find entries too heavy (should split) |
| `suggest_fusion` | Find near-duplicates (should merge) |

### Discovery
| Tool | Purpose |
|------|---------|
| `discover` | Find implicit connections in corpus geometry |
| `gaps` | Find voids — where knowledge is missing |
| `seek` | What lies beyond the edge of the sphere? |

### Health
| Tool | Purpose |
|------|---------|
| `health` | Full assessment + maintenance recommendations |
| `self_cost` | Integration measure (resonance between entries). O(N²). |
| `hunger` | Growth signal: 0=well-fed, 100=starving |
| `stability` | Per-entry binding energy |

### Maintenance
| Tool | Purpose |
|------|---------|
| `add` | Add entry with route field (use suggest_route first) |
| `fuse` | Merge two entries into synthesis |
| `fission` | Split heavy entry into focused parts |

---

## The cycle

```
1. QUERY        — auto-routes to the right specialist
2. DISCOVER     — what does the corpus imply? (not just state)
3. ACT          — work with explicit + implicit knowledge
4. VERIFY       — confirm with operator
5. UPDATE       — suggest_route(claim) → add(route=..., claim=...)
6. CONSOLIDATE  — health() → run maintenance if needed
```

### Consolidation

Janet cannot consolidate alone — she depends on the LLM to run maintenance.

Call `health()` every 5-10 interactions:
```
health() → {"action": "consolidate", "maintenance": ["improve_routes", "suggest_fusion"]}
```

Without consolidation, routes weaken and noise accumulates. self_cost warns before accuracy drops.

---

## Scale

janet-mcp uses `self_cost` (O(N²)) instead of `phi` (O(N³)) as the health metric. This means centers can grow much larger:

| Operation | 50 entries | 200 entries | 500 entries |
|---|---|---|---|
| query (orbit) | 0.1ms | 1ms | 6ms |
| self_cost | 1ms | 42ms | 550ms |
| learn_batch | 3ms | 14ms | 51ms |

**Why self_cost over phi:** self_cost is a stronger predictor of retrieval accuracy (r=0.82 vs r=0.39 for phi), is O(N²) instead of O(N³), and measures what matters in production — how well entries predict each other (internal resonance). phi remains in janet-py for research (detects phase transitions).

With federated centers, a typical setup has 20-50 entries per center. But centers can grow to 200-500 entries and still have real-time health monitoring. Total knowledge across all centers is unlimited.

---

## How it works

- **Federated centers** — each center is a specialist node, queries auto-route via GWT
- **Orbit as dynamics** — query = perturbation, orbit = relaxation to equilibrium
- **Integer arithmetic** — int32 vectors, int64 dot products, deterministic
- **DCE encoding** — distributional centroid encoding, meaning evolves with use
- **Geometric funneling** — iterated orthogonal projection, guaranteed convergence
- **Self-referential ground state** — fixed-point attractor anchors each center
- **Dual representation** — route (geometry) + claim (language)

---

## When to use janet (and when not to)

| Scenario | Janet | Alternative | Winner |
|---|---|---|---|
| Agent needs to remember decisions across weeks | ✓ persistent, self-maintaining | Markdown file (no health signal) | Janet |
| "Does the corpus know about X?" | ✓ amplitude=0 = native "no" | RAG returns top-k regardless | Janet |
| Corpus growing stale silently | ✓ self_cost detects degradation | No signal until accuracy drops | Janet |
| "How do I improve the knowledge base?" | ✓ suggest_route, improve_routes | Trial and error | Janet |
| Large-scale retrieval (10k+ docs) | ✗ O(N) per center | ChromaDB/Pinecone (optimized) | RAG |
| Paraphrase/synonym understanding | ✗ lexical co-occurrence only | Transformer embeddings | RAG |
| Adjacent topic detection | ✗ shares tokens = high amplitude | Semantic distance (gradual) | RAG |
| Zero setup, no model download | ✓ numpy only | Needs embedding model | Janet |
| Deterministic (audit/compliance) | ✓ same input = same output | Float embeddings vary | Janet |

**Janet is best for:** curated knowledge (20-500 entries per center), long-running agents, systems where "I don't know" matters, auditability, self-maintaining corpora.

**Janet is not for:** large document search, semantic paraphrasing, one-shot queries, corpora without vocabulary discipline.

---

## Limitations

1. **No semantic understanding.** Janet routes by token co-occurrence, not meaning. Queries that share vocabulary with the corpus get high amplitude even if the specific answer doesn't exist. The LLM must judge whether the returned entry actually answers the question.

2. **Adjacent queries are not detected.** amplitude=0 works for foreign queries (zero token overlap). But "Gibbs free energy" gets high amplitude if "free energy" is in the corpus. The LLM is the semantic filter — janet is the geometric router.

3. **Requires vocabulary discipline.** Route tokens must be shared deliberately between related entries. Random vocabulary breaks the geometry. suggest_route helps, but the operator/LLM must be consistent.

4. **Not a replacement for RAG.** For large-scale retrieval with paraphrases and synonyms, transformer-based RAG is better. Janet is for systems where determinism, "I don't know", and self-prescription matter more than coverage.

---

## Files

```
janet.py      — core engine (encoding, phi, orbit, generation, decay, atom_stability)
network.py    — network of nodes (GWT routing, broadcast, feedback)
mcp.py        — MCP server (all tools, auto-routing, self-prescription)
know.py       — corpus CLI
AGENTS.md     — copy into your project for agent guidance
```

---

*Specialized modules. Automatic routing. Measurable health. The geometry prescribes.*
