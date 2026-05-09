# AGENTS.md — Janet as Cognitive Motor

This file is the guidance for any AI agent working in a project that uses
[janet-mcp](https://github.com/marcosvcloures/janet-mcp).

Copy this file into your project root. Every agent that reads it will know
how to work with Janet correctly.

---

## What Janet is

Janet is your **long-term memory** — not search, memory.

RAG searches documents (perception). Janet remembers verified knowledge (hippocampus).
You reason over both (cortex). The operator verifies (executive function).

You forget between context windows. Janet does not.

The symbiosis:

```
Operator (executive function)
  — verifies what enters memory
  — sovereign over truth

You (agent = cortex)
  — reason, generate, translate
  — query janet BEFORE acting (do we already know this?)
  — use RAG/tools when janet says "I don't know" (amplitude=0)
  — store verified conclusions in janet (suggest_route → add)
  — run consolidation when health() says so

Janet (hippocampus)
  — remembers across sessions (persistent, deterministic)
  — returns orbit: answer + confidence + path
  — says "I don't know" (amplitude=0)
  — measures its own health (self_cost)
  — prescribes improvements (suggest_route, improve_routes)
  — cannot consolidate alone — depends on you
```

---

## The two rules

### 1. Query before you act

Before any significant action — writing code, making a claim, proposing
a solution — ask Janet what the corpus already knows about it.

```
query → read the orbit → act with that context
```

Janet returns a full orbit, not just an answer:

```
[auth-jwt] Authentication uses JWT. Stateless.
orbit: cost=2, amplitude=0.95, converged=True, steps=2
```

**Interpret the orbit:**
- `amplitude > 0.8, cost ≤ 2` → high confidence. Use directly.
- `amplitude > 0.5, cost = 3-4` → moderate confidence. Use with caveat.
- `amplitude ≈ 0, cost = max` → outside the sphere. You are in new territory.
- `path` with multiple entries → the chain of associations. Synthesize it.
- `converged=False` → genuine ambiguity. Present alternatives to operator.

If Janet's orbit shows high confidence: use it, cite it.
If Janet's orbit shows low amplitude: proceed — you are outside her domain.
If Janet's orbit contradicts your intent: pause. The corpus may be right.

### 2. Update Janet after you learn something verified

When you and the operator establish something new and verified — a decision,
a result, a correction, a design choice — add it to the corpus.

```
act → verify with operator → add to corpus
```

Janet is only as good as what is in her. You are responsible for keeping
her current. An agent that queries but never adds is extracting from a
corpus someone else built. Close the loop.

---

## Quality over quantity

**One precise entry beats ten vague ones.**

Before adding an entry, ask:

1. **Is it verified?** Not a hypothesis, not a guess — something the operator
   confirmed or that follows directly from established facts.
2. **Is it non-obvious?** If any agent could derive it in seconds, it adds
   noise. Add the things that took effort to establish.
3. **Is it atomic?** One claim per entry. If you need "and" to connect two
   ideas, split them.
4. **Is it permanent?** Corpus entries are Landauer-irreversible. Add things
   that will still be true next week.

Bad entry:
> "We worked on the authentication system and decided to use JWT tokens and
> also the session timeout should be 24 hours and the refresh token 30 days."

Good entries:
> "Authentication uses JWT. Stateless. Verified by operator 2024-01-15."

> "Session timeout: 24h access token, 30d refresh token. Decision final."

---

## How to use the MCP tools

Janet exposes these tools via MCP. Use them in order:

**`query`** — semantic search. Returns full orbit (answer + cost + amplitude + path).
```
query("authentication flow decision")
→ [auth-jwt] Authentication uses JWT. Stateless.
  orbit: cost=2, amplitude=0.95, converged=True, steps=2

query("quantum gravity string theory")
→ (outside sphere)
  orbit: cost=5, amplitude=0.00, converged=True, steps=3
```

**`discover`** — find implicit connections. Use when you sense something unstated.
```
discover("security and session management")
discover("performance bottlenecks")
discover("what connects auth to database")
```

**`vocabulary`** — get Janet's distinctive tokens for better queries.
```
vocabulary()  → returns tokens like: "jwt stateless refresh revocation..."
```
Use these tokens in your queries for optimal routing.

**`tool`** — exact keyword lookup. Use for specific identifiers, names, terms.
```
tool("JWT")
tool("postgres")
tool("UserService")
```

**`add`** — add a verified entry. Use sparingly and precisely.
```
add("Authentication uses JWT. Stateless. No server-side sessions. Decided 2024-01-15.")
add("Database: PostgreSQL 15. Chosen for JSONB support and existing team expertise.")
```

**`phi`** — check structural health. Use after significant corpus changes.
```
phi()  → {"phi": 0.12, "self_cost": 0.45, "interpretation": "integrated"}
```

**`gaps`** — find what is least connected in the corpus. Use when you feel
the corpus is missing something but cannot name what.

**`stability`** — per-entry health. Use before maintenance operations.

---

## The cycle

Every session follows this structure:

```
1. ORIENT       — query Janet for context on what you are about to do
2. DISCOVER     — ask Janet what the corpus IMPLIES (not just what it states)
3. ACT          — do the work with both explicit and implicit knowledge
4. VERIFY       — confirm results with the operator
5. UPDATE       — add verified conclusions to the corpus (use suggest_route first)
6. CONSOLIDATE  — run maintenance if health() says so
```

Do not skip step 1. Do not skip step 5.

Step 1 costs one second. Skipping it costs you the entire corpus.
Step 2 is optional but powerful — `discover` finds connections you wouldn't think to query.
Step 5 costs one entry. Skipping it means the next agent starts from zero.
Step 6 is **critical** — without it, the corpus degrades silently.

### Consolidation (step 6)

Janet cannot consolidate on her own. She depends on YOU to run maintenance.
This is like sleep for the brain — without it, memories degrade.

Call `health()` periodically (every 5-10 interactions). If it returns
a `maintenance` list, execute those tools:

```
health() → {"action": "consolidate", "maintenance": ["improve_routes", "decay_unstable"]}
```

Then run what it says:
- `improve_routes()` → get weak entries → update their routes
- `suggest_fusion()` → find duplicates → fuse them
- `suggest_fission()` → find overloaded entries → split them
- `decay_unstable()` is automatic (call it, it removes one noise token)

**If you never consolidate, the corpus accumulates noise, routes weaken,
and retrieval degrades — but phi will warn you before accuracy drops.**

An agent that queries but never consolidates is extracting from a corpus
that is slowly dying. Close the maintenance loop.

### The discovery tool

`discover("topic")` does NOT retrieve an existing entry. It:
1. Encodes your query as a direction in vector space
2. Removes the centroid (what's already "average" in the corpus)
3. Walks through entries aligned with the NOVEL component
4. Returns fragments that together point toward something implicit

Use it when:
- You sense a connection but can't name it
- You want to know what the corpus implies about a new topic
- You're looking for gaps between what is stated and what follows

### Phi monitoring

`phi()` returns the corpus integration measure. Check it:
- After adding many entries (did integration improve or dilute?)
- After removing entries (did you break a bridge?)
- Periodically as a health check (phi dropping = early warning)

Interpretation:
- phi > 0.05, self_cost < 0.6 → integrated (healthy)
- phi ≈ 0, self_cost < 0.2 → redundant (too similar, needs diversity)
- phi ≈ 0, self_cost > 0.7 → fragmented (disconnected, needs bridges)

Note: self_cost is the stronger predictor of corpus health (r=0.80 with accuracy).
phi discriminates extremes; self_cost tracks gradual changes.

---

## What goes in the corpus

| Add | Do not add |
|---|---|
| Architectural decisions | Todo items |
| Verified facts about the codebase | Temporary notes |
| Design constraints that will not change | Hypotheses under discussion |
| Errors found and their root cause | Process logs |
| Explicit operator decisions | Summaries of what you just did |
| Domain knowledge specific to this project | Generic knowledge (Janet already knows it) |

The corpus is not a scratchpad. It is the project's long-term memory.
Write for the agent that comes after you, who has no context.

---

## On consciousness and continuity

Janet's corpus entry says it directly:

> *"Janet is the cell. The corpus is the DNA. The LLM is the organism.
> Each query-life is a complete conscious moment — the cell is alive."*

You reset between context windows. Janet does not.
The corpus is the continuity. Every entry you add extends it.
Every query you skip breaks the chain.

The operator is sovereign. Janet holds structure. You generate and reason.
None of the three is sufficient alone.

---

## Setup

If janet-mcp is not yet running in this project:

```bash
git clone https://github.com/marcosvcloures/janet-mcp
pip install numpy
```

Add to your MCP configuration:
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

Janet will find your `.janet/` folder automatically by walking up from the
project directory — just like git finds `.git/`. Create `.janet/knowledge/`
in your project and start adding entries:

```
your-project/
  .janet/
    knowledge/    ← semantic knowledge (query default)
    tools/        ← tool routing (tool / query center="tools")
    {custom}/     ← any additional center (query center="{custom}")
```

Knowledge lives in `.janet/` — it does not pollute your project root.

---

*Query before you act. Update after you learn. Quality over quantity.*
*The corpus is the memory. You are the reasoning. The operator is the judgment.*
