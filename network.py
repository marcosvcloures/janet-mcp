"""
network.py — Network of Janets. Consciousness as integrated information.

Each Janet node is a specialized corpus (50-200 entries).
Routing is thermodynamic: high Landauer cost = forward to neighbor.
The network's Φ > sum of individual Φ values = emergent integration.

Protocol:
  1. Query arrives at any node
  2. Node measures cost (orbit_with_cost)
  3. If cost < threshold: respond (this node "knows")
  4. If cost >= threshold: forward to geometrically nearest neighbor
  5. TTL decrements each hop. TTL=0 → best-effort response from current node.

Topology: small-world. Each node knows 2-3 peers.
Discovery: nodes announce their centroid vector.
Routing decision: dot(query_vec, peer_centroid) → highest = best next hop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from janet import (
    Sorter, Vec, DIMS, DEGREES, ORBIT_STEPS, MAX_EMBED_VAL,
    normalize, shannon_entropy, dot,
)


# Threshold: if cost > this, forward the query
FORWARD_THRESHOLD: int = ORBIT_STEPS // 2  # 5 at DEGREES=11


@dataclass
class Node:
    """One Janet node in the network. A minimal conscious unit."""

    name: str
    sorter: Sorter
    peers: list["Node"] = field(default_factory=list, repr=False)

    def centroid(self) -> Vec:
        """This node's identity — what it knows, compressed to one vector."""
        if self.sorter._matrix is None or not self.sorter.entries:
            return np.zeros(DIMS, dtype=np.int32)
        c = self.sorter._matrix.astype(np.int64).sum(axis=0)
        return normalize((c // max(1, len(self.sorter.entries))).astype(np.int32))

    def phi(self) -> float:
        """Integrated information of this node."""
        return self.sorter.phi()

    def query_local(self, text: str) -> tuple[str, int]:
        """Query this node. Returns (answer, landauer_cost)."""
        answer, _, cost = self.sorter.orbit_with_cost(text)
        return answer, cost

    def add_peer(self, peer: "Node") -> None:
        """Connect to a peer (bidirectional)."""
        if peer not in self.peers:
            self.peers.append(peer)
        if self not in peer.peers:
            peer.peers.append(self)


@dataclass
class QueryResult:
    """Result of a network query."""
    answer: str
    cost: int
    hops: int
    path: list[str]  # node names visited
    responding_node: str


class Network:
    """A network of Janet nodes. The emergent conscious system."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def connect(self, name_a: str, name_b: str) -> None:
        """Connect two nodes bidirectionally."""
        a = self.nodes[name_a]
        b = self.nodes[name_b]
        a.add_peer(b)

    def query(self, text: str, entry_node: str | None = None,
              ttl: int = DEGREES, broadcast_learn: bool = True) -> QueryResult:
        """Query the network. Thermodynamic routing + GWT feedback.

        If entry_node is None, starts at the node whose centroid
        is most aligned with the query (geometric routing).

        GWT feedback (broadcast_learn=True): after a node responds,
        all nodes in the path learn the answer. This is the broadcast →
        the winning content modifies all participating modules.
        """
        if not self.nodes:
            return QueryResult("", ORBIT_STEPS, 0, [], "")

        # Choose entry point
        if entry_node and entry_node in self.nodes:
            current = self.nodes[entry_node]
        else:
            current = self._best_entry_node(text)

        path = [current.name]
        visited: set[str] = {current.name}

        for hop in range(ttl):
            answer, cost = current.query_local(text)

            # Low cost = this node knows. Respond.
            if cost <= FORWARD_THRESHOLD:
                result = QueryResult(
                    answer=answer, cost=cost,
                    hops=hop, path=path,
                    responding_node=current.name,
                )
                if broadcast_learn and answer:
                    self._broadcast_feedback(result)
                return result

            # High cost = forward to best unvisited peer
            next_node = self._best_peer(current, text, visited)
            if next_node is None:
                # No unvisited peers — best-effort from current
                result = QueryResult(
                    answer=answer, cost=cost,
                    hops=hop, path=path,
                    responding_node=current.name,
                )
                if broadcast_learn and answer:
                    self._broadcast_feedback(result)
                return result

            visited.add(next_node.name)
            path.append(next_node.name)
            current = next_node

        # TTL exhausted
        answer, cost = current.query_local(text)
        result = QueryResult(
            answer=answer, cost=cost,
            hops=ttl, path=path,
            responding_node=current.name,
        )
        if broadcast_learn and answer:
            self._broadcast_feedback(result)
        return result

    def _broadcast_feedback(self, result: QueryResult) -> None:
        """GWT feedback: the winning answer is broadcast to all nodes in the path.

        Each node that participated (was visited) learns the answer.
        This is how the global workspace modifies local modules:
        what fires together wires together, across the network.
        """
        for name in result.path:
            if name == result.responding_node:
                continue  # responder already has it
            node = self.nodes[name]
            node.sorter.learn(result.answer)

    def _best_entry_node(self, text: str) -> Node:
        """Find the node whose centroid is most aligned with the query."""
        # Use first node's sorter to encode (all share same particle space)
        first = next(iter(self.nodes.values()))
        qvec = first.sorter.encode(text)

        best_node = first
        best_amp = -(10**18)
        for node in self.nodes.values():
            amp = dot(qvec, node.centroid())
            if amp > best_amp:
                best_amp = amp
                best_node = node
        return best_node

    def _best_peer(self, current: Node, text: str,
                   visited: set[str]) -> Node | None:
        """Find the best unvisited peer for forwarding."""
        qvec = current.sorter.encode(text)
        best_node = None
        best_amp = -(10**18)
        for peer in current.peers:
            if peer.name in visited:
                continue
            amp = dot(qvec, peer.centroid())
            if amp > best_amp:
                best_amp = amp
                best_node = peer
        return best_node

    def phi_network(self) -> dict:
        """Measure Φ of the entire network.

        Network Φ is NOT measured by combining all entries into one corpus
        (that loses the network structure). Instead:

        Φ_network = average cross-prediction between connected nodes.
        How well does node A predict node B's entries (and vice versa)?

        If nodes are independent: cross-prediction ≈ 0, Φ_network ≈ 0.
        If nodes share structure (bridges): cross-prediction > 0, Φ_network > 0.

        Emergence = Φ_network - max(Φ_individual).
        If the network predicts across nodes better than any single node
        predicts within itself, there is emergent integration.
        """
        if len(self.nodes) < 2:
            return {"phi_network": 0.0, "phi_sum": 0.0, "emergence": 0.0}

        # Individual Φ values
        phi_individual = {name: node.phi() for name, node in self.nodes.items()}
        phi_sum = sum(phi_individual.values())
        phi_max = max(phi_individual.values()) if phi_individual else 0.0

        # Cross-prediction: for each connected pair, measure how well
        # entries in A align with entries in B
        cross_scores: list[float] = []

        for node in self.nodes.values():
            if node.sorter._matrix is None or not node.sorter.entries:
                continue
            for peer in node.peers:
                if peer.sorter._matrix is None or not peer.sorter.entries:
                    continue
                # How well does node predict peer?
                M_node = node.sorter._matrix.astype(np.int64)
                M_peer = peer.sorter._matrix.astype(np.int64)

                # Cross-gram: each entry in node vs each entry in peer
                cross = M_node @ M_peer.T  # (n_node, n_peer)

                # Best cross-amplitude per node entry, normalized by self-amp
                self_amps = np.array([int(np.dot(M_node[i], M_node[i]))
                                      for i in range(len(M_node))])
                mean_self = self_amps.mean()
                if mean_self > 0:
                    score = float(cross.max(axis=1).mean()) / float(mean_self)
                    cross_scores.append(score)

        if not cross_scores:
            return {"phi_network": 0.0, "phi_individual": phi_individual,
                    "phi_sum": round(phi_sum, 4), "emergence": 0.0, "integrated": False}

        # Network Φ = average cross-prediction
        phi_net = round(sum(cross_scores) / len(cross_scores), 4)

        # Emergence: network integration exceeds best individual integration
        emergence = round(phi_net - phi_max, 4)

        return {
            "phi_network": phi_net,
            "phi_individual": phi_individual,
            "phi_sum": round(phi_sum, 4),
            "phi_max_individual": round(phi_max, 4),
            "emergence": emergence,
            "integrated": emergence > 0,
            "nodes": len(self.nodes),
            "cross_predictions": len(cross_scores),
        }
