#!/usr/bin/env python3
"""
server.py — Janet SSE MCP server.

One process. One world. Built from YAML. Served in O(1).

The YAML files are the Janet network embodied:
  - Nodes  = entries (id, claim, tags, confidence)
  - Edges  = related: links (co-learned → pulled together in embedding space)
  - Corpus = rebuilt from YAML at startup, never written to disk

The server:
  1. Scans knowledge/ dir → loads all YAML → builds matrix once (O(N))
  2. Watches for YAML changes → incremental reload (O(changed domain))
  3. Serves all agent sessions from shared in-memory corpus
  4. Answers queries in O(1) — one matrix @ vec dot product

Janet does not save geometry. She observes what is in front of her.
The YAML is the world. The matrix is the observation. The query is the collapse.

Usage:
  python3 server.py                          # serve on 0.0.0.0:7447
  python3 server.py --port 8080              # custom port
  python3 server.py --knowledge /path/to/kb  # custom knowledge dir

MCP config (.mcp.json / opencode.jsonc):
  {
    "mcpServers": {
      "janet": {
        "type": "sse",
        "url": "http://localhost:7447/sse"
      }
    }
  }
"""

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Empty, Queue
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from mcp import (
    KnowledgeStore, exec_bash, exec_edit, exec_grep, exec_read, exec_write,
    TOOLS,
)

# ── World ─────────────────────────────────────────────────────────────────

class World:
    """The world. Built from YAML. Observable but not storable.

    One instance per server process. All sessions share it.
    YAML files are watched for changes — reload is automatic.
    """

    def __init__(self, knowledge_dir: Path, cwd: str):
        self.knowledge_dir = knowledge_dir
        self.cwd           = cwd
        self.store         = KnowledgeStore(knowledge_dir)
        self._lock         = threading.RLock()
        self._mtimes:  dict[str, float] = {}
        self._snapshot_mtimes()
        self._start_watcher()
        n_domains = len({e.get("domain","?") for e in self.store.entries})
        print(f"[janet] world built — {n_domains} domains, "
              f"{self.store.total_entries()} entries", flush=True)

    def _snapshot_mtimes(self):
        if not self.knowledge_dir.exists():
            return
        for pat in ("*.jsonl",):
            for f in self.knowledge_dir.rglob(pat):
                self._mtimes[str(f)] = f.stat().st_mtime

    def _start_watcher(self):
        """Watch all knowledge files (.jsonl). Reload on any change."""
        def watch():
            while True:
                time.sleep(1)
                if not self.knowledge_dir.exists():
                    continue
                changed = []
                for pat in ("*.jsonl",):
                    for f in self.knowledge_dir.rglob(pat):
                        mtime = f.stat().st_mtime
                        if self._mtimes.get(str(f)) != mtime:
                            self._mtimes[str(f)] = mtime
                            changed.append(f)
                if changed:
                    with self._lock:
                        self.store.reload()
                    n_domains = len({e.get("domain","?") for e in self.store.entries})
                    print(f"[janet] reloaded {len(changed)} files — "
                          f"{n_domains} domains, {self.store.total_entries()} entries",
                          flush=True)
        threading.Thread(target=watch, daemon=True).start()

    def query(self, q: str) -> str:
        with self._lock:
            return self.store.query(q)

    def add(self, domain: str, entry: dict) -> str:
        with self._lock:
            result = self.store.add(domain, entry)
            self._snapshot_mtimes()
            return result

    def stats(self) -> str:
        with self._lock:
            n_domains = len({e.get("domain","?") for e in self.store.entries})
            return (f"{n_domains} domains | "
                    f"{self.store.total_entries()} entries\n\n"
                    + self.store.stats())

    def reload(self) -> str:
        with self._lock:
            self.store.reload()
            self._snapshot_mtimes()
            n_domains = len({e.get("domain","?") for e in self.store.entries})
            return (f"reloaded — {n_domains} domains | "
                    f"{self.store.total_entries()} entries")


# ── Session ───────────────────────────────────────────────────────────────

class Session:
    """One agent connection. Shares the world."""

    def __init__(self, sid: str, world: World):
        self.sid   = sid
        self.world = world
        self.queue: Queue = Queue()  # server→client SSE events

    def send(self, event: str, data: Any):
        self.queue.put({"event": event, "data": data})

    def handle_request(self, req: dict) -> dict | None:
        method = req.get("method", "")
        rid    = req.get("id")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "janet", "version": "6.0-sse"}
                }
            }

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

        if method == "tools/call":
            name  = params.get("name", "")
            args  = params.get("arguments", {})
            cwd   = self.world.cwd

            if name == "query":
                result = self.world.query(args.get("q", ""))
                if not result:
                    result = "(outside knowledge sphere — use 'add' to create this entry)"

            elif name == "add":
                entry = {k: v for k, v in {
                    "id":         args.get("id", ""),
                    "claim":      args.get("claim", ""),
                    "domain":     args.get("domain", ""),
                    "tags":       args.get("tags", []),
                    "confidence": args.get("confidence", ""),
                    "source":     args.get("source", ""),
                }.items() if v}
                result = self.world.add(args.get("domain", "general"), entry)

            elif name == "gaps":
                result = json.dumps(
                    self.world.store.gaps(args.get("pair_idx", 0)),
                    indent=2, ensure_ascii=False)

            elif name == "seek":
                result = json.dumps(self.world.store.seek(), indent=2, ensure_ascii=False)

            elif name == "stability":
                result = json.dumps(self.world.store.stability(args.get("k", 5)), indent=2, ensure_ascii=False)

            elif name == "fuse":
                r = self.world.store.fuse(args.get("id_a",""), args.get("id_b",""))
                result = json.dumps(r, indent=2, ensure_ascii=False) if r else "(no result)"

            elif name == "fission":
                r = self.world.store.fission(args.get("id",""), execute=args.get("execute", False))
                result = json.dumps(r, indent=2, ensure_ascii=False) if r else "(no result)"

            elif name == "hunger":
                result = json.dumps(self.world.store.hunger(), indent=2)

            elif name == "health":
                result = json.dumps(self.world.store.health(), indent=2)

            elif name == "tool":
                from mcp import _tool_query
                result = _tool_query(args.get("q", ""), cwd)

            elif name == "stats":
                result = self.world.stats()

            elif name == "reload":
                result = self.world.reload()

            elif name == "write":
                result = exec_write(args.get("path",""), args.get("content",""), cwd)
                # YAML write → world auto-reloads via watcher

            elif name == "edit":
                result = exec_edit(args.get("path",""), args.get("old_string",""),
                                   args.get("new_string",""), cwd)

            elif name == "bash":
                result = exec_bash(args.get("command",""), cwd)

            elif name == "read":
                result = exec_read(args.get("path",""), cwd)

            elif name == "grep":
                result = exec_grep(args.get("pattern",""), args.get("path","."), cwd)

            else:
                result = f"unknown tool: {name}"

            return {
                "jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": result}]}
            }

        return {
            "jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"unknown method: {method}"}
        }


# ── HTTP / SSE handler ────────────────────────────────────────────────────

class JanetHandler(BaseHTTPRequestHandler):

    world:    World
    sessions: dict  # sid → Session

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/sse" or self.path.startswith("/sse?"):
            self._handle_sse()
        elif self.path == "/health":
            self._json(200, {"status": "ok",
                             "entries": self.world.total_entries()
                                         if hasattr(self.world, 'total_entries')
                                         else self.world.store.total_entries()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/message" or self.path.startswith("/message?"):
            self._handle_message()
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _handle_sse(self):
        """Open SSE stream for one agent session."""
        sid     = str(uuid.uuid4())
        session = Session(sid, self.world)
        self.sessions[sid] = session

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()

        # Send session endpoint
        endpoint = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}/message?sessionId={sid}"
        self._sse_write("endpoint", endpoint)

        print(f"[janet] session {sid[:8]} connected", flush=True)

        # Stream events until client disconnects
        try:
            while True:
                try:
                    evt = session.queue.get(timeout=15)
                    self._sse_write(evt["event"], json.dumps(evt["data"]))
                except Empty:
                    self._sse_write("ping", "")  # keep-alive
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.sessions.pop(sid, None)
            print(f"[janet] session {sid[:8]} disconnected", flush=True)

    def _sse_write(self, event: str, data: str):
        try:
            msg = f"event: {event}\ndata: {data}\n\n"
            self.wfile.write(msg.encode())
            self.wfile.flush()
        except Exception:
            raise BrokenPipeError

    def _handle_message(self):
        """Receive JSON-RPC from agent, route to session, reply via SSE."""
        from urllib.parse import urlparse, parse_qs
        sid = parse_qs(urlparse(self.path).query).get("sessionId", [""])[0]

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        # Acknowledge immediately
        self.send_response(202)
        self._cors()
        self.end_headers()

        try:
            req     = json.loads(body)
            session = self.sessions.get(sid)
            if not session:
                return
            resp = session.handle_request(req)
            if resp is not None:
                session.send("message", resp)
        except Exception as e:
            print(f"[janet] message error: {e}", flush=True)


# ── Server ────────────────────────────────────────────────────────────────

def make_handler(world: World, sessions: dict):
    class Handler(JanetHandler):
        pass
    Handler.world    = world
    Handler.sessions = sessions
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Janet SSE MCP server")
    parser.add_argument("--port",      type=int, default=7447)
    parser.add_argument("--host",      default="0.0.0.0")
    parser.add_argument("--knowledge", default=None,
                        help="Path to knowledge dir (default: $JANET_DIR/knowledge or ./knowledge)")
    args = parser.parse_args()

    cwd           = os.environ.get("JANET_DIR") or os.getcwd()
    knowledge_dir = Path(args.knowledge) if args.knowledge else Path(cwd) / "knowledge"

    print(f"[janet] cwd={cwd}", flush=True)
    print(f"[janet] knowledge={knowledge_dir}", flush=True)

    world    = World(knowledge_dir, cwd)
    sessions: dict = {}
    handler  = make_handler(world, sessions)

    server = HTTPServer((args.host, args.port), handler)
    print(f"[janet] SSE MCP server — http://{args.host}:{args.port}/sse", flush=True)
    print(f"[janet] health         — http://{args.host}:{args.port}/health", flush=True)
    print(f"[janet] MCP config:", flush=True)
    print(f'[janet]   {{"type":"sse","url":"http://localhost:{args.port}/sse"}}', flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[janet] shutdown", flush=True)


if __name__ == "__main__":
    main()
