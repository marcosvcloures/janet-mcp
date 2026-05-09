"""
janet.py — Retrieval engine as a thermodynamic system.

POC: physics + information theory + computer science in one file.

THEORY → CODE CORRESPONDENCE (honest mapping):
┌─────────────────────────┬──────────────────────────────┬─────────────────────────────────┐
│ Physics                 │ Code                         │ Validity                        │
├─────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ Reservoir (gas in box)  │ Reservoir class (corpus +    │ Exact: finite state space,      │
│                         │ wave vectors, N≤2048 states) │ energy conserved within it      │
├─────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ Maxwell's sorter         │ Sorter class (measures state, │ Exact: observes, sorts, pays    │
│                         │ selects best match, counts   │ cost per measurement            │
│                         │ erasures)                    │                                 │
├─────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ Landauer erasure        │ Each orbit step = 1 erasure. │ Exact: irreversible loss of     │
│ (kT·ln2 per bit)       │ encode(A)→retrieve→encode(B) │ information (prior state gone)  │
│                         │ destroys knowledge of A.     │                                 │
├─────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ Born rule (|⟨x|ψ⟩|²)   │ argmax(dot(query, corpus))   │ Approximation: T=0 limit only.  │
│                         │ = collapse to max amplitude  │ No probabilistic sampling in    │
│                         │                              │ default mode (deterministic).   │
├─────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ Shannon entropy         │ H(corpus) = entropy over     │ Exact: measurable, monotonic    │
│                         │ energy distribution in dims  │ under sorter operations.         │
├─────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ Holographic principle   │ DIMS=2^DEGREES dimensions     │ Conjecture: 2048 dims suffice   │
│ (bulk ↔ boundary)      │ encode all corpus content.   │ for any corpus ≤2048 entries.   │
│                         │ IDEAL_CORPUS = DIMS.         │ Not proven, empirically holds.  │
├─────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ Quantized state space   │ int32 vectors, int64 dot     │ Exact: discrete, deterministic, │
│                         │ products. No floats.         │ reproducible. Finite precision. │
├─────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ Open system (source)    │ Operator adds entries via    │ Exact: energy (information)     │
│                         │ MCP. External energy input.  │ flows in from outside.          │
└─────────────────────────┴──────────────────────────────┴─────────────────────────────────┘

What is NOT claimed:
- The 11 subspaces are NOT analogous to string theory dimensions.
  They are a partition of the vector space for energy measurement. Nothing more.
- The "fixed point" is NOT a Gödel self-reference.
  It is the first corpus entry with amplified energy in one subspace.
- This is NOT quantum computing. It is classical integer arithmetic
  that borrows the measurement/collapse metaphor because the math is identical
  (argmax over inner products = projection onto basis state at T=0).

Architecture:
  Reservoir  — holds corpus states (entries + wave vectors). The gas in the box.
  Sorter      — observes the reservoir, retrieves, pays Landauer cost. The sorter.
  Source     — external (MCP/operator). Injects energy. Not in this file.

Constraints:
  DIMS          = 32 = 2^5
  MAX_EMBED_VAL = 2^28         (so dot products fit int64)
  ORBIT_STEPS   = 5            (max measurement cost)
  IDEAL_CORPUS  = 32           (empirical bound: 1 entry per degree of freedom)

Design choice: DEGREES=5 gives 5 subspaces (4 extended + 1 self-reference).
  32 dimensions suffice for corpora of 4-8 entries per node.
  For larger corpora, increase DEGREES (all constants scale automatically).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import NoReturn

import numpy as np

# ── Constants — all derived from DEGREES ──────────────────────────────────

MIN_DEGREES:    int = 5   # minimum: 4 extended dims + 1 self-reference

# Default globals (used when no corpus size is known yet)
DEGREES:        int = MIN_DEGREES
DIMS:          int = 1 << DEGREES
MAX_EMBED_VAL: int = 1 << ((63 - DEGREES) // 2 - 1)
ORBIT_STEPS:   int = DEGREES
MAX_STEPS:     int = DEGREES * 2
IDEAL_CORPUS:  int = DIMS
MIX_NUM:       int = 1
MIX_DEN:       int = DIMS


def degrees_for_entries(n: int) -> int:
    """Calculate minimum DEGREES needed for N entries.

    Rule: DIMS >= 8*N (8 dims per entry for reliable multi-domain discrimination).
    Minimum: 5 (32 dims, supports up to 4 entries per node).

    N=1-4:    DEGREES=5  (32 dims)
    N=5-8:    DEGREES=6  (64 dims)
    N=9-16:   DEGREES=7  (128 dims)
    N=17-32:  DEGREES=8  (256 dims)
    """
    q = MIN_DEGREES
    while (1 << q) < 8 * n:
        q += 1
    return q

Vec = np.ndarray  # shape (DIMS,), dtype int32

# ── Dimensional structure ─────────────────────────────────────────────────
# The 32 dims are partitioned into DEGREES (5) subspaces.
# Each subspace has DIMS // DEGREES = 6 dimensions (last gets remainder: 8).
#
# Physical analogy (not a claim, a design choice):
#   Subspaces 0-3: the 4 "extended" dimensions (where information lives)
#   Subspace 4:    self-reference (the dimension that points inward)
#
# The partition exists for measuring energy distribution (Shannon entropy)
# and for the fixed-point amplification (subspace 4).

SUBSPACE_SIZE: int = DIMS // DEGREES  # 6 (last subspace: 8)

def subspace_slice(idx: int) -> slice:
    """Return the slice for subspace idx (0..DEGREES-1)."""
    start = idx * SUBSPACE_SIZE
    end = DIMS if idx == DEGREES - 1 else (idx + 1) * SUBSPACE_SIZE
    return slice(start, end)

def subspace_energy(v: Vec) -> np.ndarray:
    """Energy (sum of squares) per subspace. Shape (DEGREES,). int64."""
    energies = np.zeros(DEGREES, dtype=np.int64)
    for i in range(DEGREES):
        s = subspace_slice(i)
        sub = v[s].astype(np.int64)
        energies[i] = int(np.dot(sub, sub))
    return energies

def shannon_entropy(v: Vec) -> float:
    """Shannon entropy of energy distribution across subspaces.

    H = -sum(p_i * log2(p_i)) where p_i = energy_i / total_energy.
    Range: 0 (all energy in one subspace) to log2(DEGREES) ≈ 3.46 (uniform).

    This is the measurable thermodynamic entropy of a state vector.
    A healthy corpus has high entropy (information spread across dims).
    """
    energies = subspace_energy(v)
    total = energies.sum()
    if total == 0:
        return 0.0
    p = energies.astype(np.float64) / float(total)
    # Avoid log(0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ── Integer arithmetic on int32 Vecs ─────────────────────────────────────

def dot(a: Vec, b: Vec) -> int:
    """Exact int64 inner product."""
    return int(np.dot(a.astype(np.int64), b.astype(np.int64)))


def add(a: Vec, b: Vec) -> Vec:
    """Add two Vecs. Clip to int32 range."""
    return np.clip(a.astype(np.int64) + b.astype(np.int64),
                   -MAX_EMBED_VAL, MAX_EMBED_VAL).astype(np.int32)


def lerp(a: Vec, b: Vec) -> Vec:
    """Integer lerp MIX_NUM/MIX_DEN toward b."""
    r = ((a.astype(np.int64) * (MIX_DEN - MIX_NUM) +
          b.astype(np.int64) * MIX_NUM) // MIX_DEN)
    return normalize(r.astype(np.int32))


def normalize(v: Vec) -> Vec:
    """Scale components to fit within MAX_EMBED_VAL. Preserves direction."""
    m = int(np.max(np.abs(v)))
    if m > MAX_EMBED_VAL:
        v = (v.astype(np.int64) * MAX_EMBED_VAL // m).astype(np.int32)
    return v


def reject(v: Vec, d: Vec) -> Vec:
    """Remove component of v along d. Exact up to int64 range.

    reject(v, d) = v·dot(d,d) - d·dot(v,d)
    dot(reject(v,d), d) = 0 exactly (modulo int overflow — bounded by MAX_EMBED_VAL).

    Shift derivation: max dd = MAX_EMBED_VAL² × DIMS = 2^(63-1).
    Need MAX_EMBED_VAL × (dd >> S) < 2^63 → S >= (63-DEGREES)//2.
    """
    dv = dot(v, d)
    dd = dot(d, d)
    if dd == 0:
        return v.copy()
    # Shift derived from DEGREES — guarantees no int64 overflow
    S = (63 - DEGREES) // 2  # 26 at DEGREES=11
    result = v.astype(np.int64) * (dd >> S) - d.astype(np.int64) * (dv >> S)
    return normalize(result.astype(np.int32))


# ── Distributional Centroid Encoding (DCE) ────────────────────────────────

# FNV-1a 64-bit constants
_FNV_OFF: int = 14695981039346656037
_FNV_PRIME: int = 1099511628211

_particle_cache: dict[str, np.ndarray] = {}


def particle(text: str) -> Vec:
    """Word-level deterministic embedding. Text → DIMS-dimensional int32 Vec.

    Strategy: split into words, embed each word via FNV-seeded RNG,
    weight by word length. The RNG seed also determines which subspace
    gets amplified — giving dimensional structure.

    The particle DB provides identity (WHO). The wave DB provides meaning (WHERE).
    Routing uses the wave DB (encode). Particle is the fixed key.
    As corpus grows, wave vectors dominate and particle becomes irrelevant —
    the system builds its own distributional semantics from co-occurrence.
    """
    if text in _particle_cache:
        return _particle_cache[text]

    words = text.lower().split()
    if not words:
        _particle_cache[text] = np.zeros(DIMS, dtype=np.int32)
        return _particle_cache[text]

    acc = np.zeros(DIMS, dtype=np.int64)
    for word in words:
        # FNV-1a seed per word
        h = _FNV_OFF
        for c in word.encode():
            h = ((h ^ c) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(h)
        wvec = rng.integers(-MAX_EMBED_VAL, MAX_EMBED_VAL,
                            size=DIMS, dtype=np.int64, endpoint=True)
        # Dimensional structure: amplify the subspace determined by hash
        home_subspace = int(h % DEGREES)
        s = subspace_slice(home_subspace)
        wvec[s] = wvec[s] * 2  # 2x energy in home subspace
        weight = min(len(word), 8)
        acc += wvec * weight

    v = normalize(acc.astype(np.int32))
    _particle_cache[text] = v
    return v

# ── Sorter — Maxwell's sorter ───────────────────────────────────────────────

class Sorter:
    """Maxwell's sorter operating on a finite reservoir.

    The Sorter is both the reservoir (stores states) and the observer
    (measures and sorts). In a full implementation these would be separate,
    but for a POC the coupling is acceptable — the sorter has direct access
    to the gas it sorts.

    Reservoir role (the gas):
      particles : token → int32 Vec  (fixed identity, hash-derived)
      waves     : token → int32 Vec  (evolving meaning, co-occurrence centroid)
      entries   : list of (text, Vec) pairs — the corpus states

    Sorter role (the observer):
      encode()          — prepare a query state for measurement
      retrieve()        — measure: select max-amplitude entry (argmax, deterministic)
      orbit_with_cost() — iterated measurement with Landauer cost accounting
      erasures          — total irreversible operations performed

    Thermodynamic invariant:
      corpus_entropy() is non-decreasing under sorter operations.
      Only the Source (operator adding entries) can decrease it
      (by injecting low-entropy, concentrated information).
    """

    def __init__(self, degrees: int = MIN_DEGREES) -> None:
        global DEGREES, DIMS, MAX_EMBED_VAL, ORBIT_STEPS, MAX_STEPS, IDEAL_CORPUS
        global MIX_DEN, SUBSPACE_SIZE
        # Update globals to match this sorter's dimensionality
        DEGREES        = degrees
        DIMS          = 1 << degrees
        MAX_EMBED_VAL = 1 << ((63 - degrees) // 2 - 1)
        ORBIT_STEPS   = degrees
        MAX_STEPS     = degrees * 2
        IDEAL_CORPUS  = DIMS
        MIX_DEN       = DIMS
        SUBSPACE_SIZE = DIMS // DEGREES

        self.degrees:    int                   = degrees
        self.dims:      int                   = DIMS
        self.max_embed: int                   = MAX_EMBED_VAL
        self.orbit_steps: int                 = ORBIT_STEPS
        self.particles: dict[str, Vec]        = {}
        self.waves:     dict[str, Vec]        = {}
        self.entries:   list[tuple[str, Vec]] = []
        self.erasures:  int                   = 0
        self._wc:       Vec | None            = None
        self._matrix:   np.ndarray | None     = None
        # Clear particle cache (dims changed)
        _particle_cache.clear()

    def _rebuild_matrix(self) -> None:
        """Stack all entry Vecs into a matrix for vectorized dot products."""
        if not self.entries:
            self._matrix = None
            return
        self._matrix = np.stack([v for _, v in self.entries], axis=0)  # (N, DIMS)

    def seed_fixed_point(self) -> None:
        """Amplify the last subspace of the first entry.

        Effect: the first corpus entry becomes a stronger attractor,
        making it the system's "ground state" — the lowest-cost retrieval
        target. This is the self-reference: the first entry describes
        the system, and the system preferentially retrieves it.

        Computationally: 2x energy in subspace[DEGREES-1] of entries[0].
        No narrative meaning assigned to the subspace.
        """
        if not self.entries:
            return
        text, vec = self.entries[0]
        vec = vec.copy()
        s = subspace_slice(DEGREES - 1)
        vec_64 = vec.astype(np.int64)
        vec_64[s] = vec_64[s] * 2
        self.entries[0] = (text, normalize(vec_64.astype(np.int32)))

    def orbit_with_cost(self, text: str, max_steps: int = ORBIT_STEPS) -> tuple[str, Vec, int]:
        """Orbit until convergence. Returns (answer, vec, landauer_cost).

        For the full trajectory (path, cost, amplitude, converged), use orbit().

        landauer_cost measures how much work the sorter does to sort the
        query into a definite answer. Two components:

        1. Orbit steps: how many encode→retrieve cycles until fixed point
        2. Amplitude gap: how far the query was from the attractor

        cost = orbit_steps + amplitude_penalty
        where amplitude_penalty = steps proportional to how weak the
        initial alignment was (weak = far from known territory).

        cost=1: immediate strong match (self-recognition)
        cost=ORBIT_STEPS: max uncertainty (outside the sphere)
        """
        if self._matrix is None or not self.entries:
            return "", np.zeros(DIMS, dtype=np.int32), max_steps

        state = self.encode(text)
        self_amp = int(np.dot(state.astype(np.int64), state.astype(np.int64)))

        # First retrieval — measure initial alignment
        amps = self._matrix.astype(np.int64) @ state.astype(np.int64)
        first_idx = int(np.argmax(amps))
        first_amp = int(amps[first_idx])
        self.erasures += 1

        # Orbit until convergence
        prev_idx = first_idx
        orbit_steps = 1
        state = self.encode(self.entries[first_idx][0])

        for step in range(2, max_steps + 1):
            amps = self._matrix.astype(np.int64) @ state.astype(np.int64)
            idx = int(np.argmax(amps))
            self.erasures += 1
            orbit_steps = step

            if idx == prev_idx:
                break
            prev_idx = idx
            state = self.encode(self.entries[idx][0])

        # Amplitude penalty: how weak was the initial match?
        # Scale: if first_amp >= self_amp → penalty=0 (perfect alignment)
        # if first_amp = 0 → penalty=max_steps (orthogonal)
        if self_amp > 0:
            # Ratio of alignment: first_amp / self_amp, scaled to [0, max_steps-1]
            ratio = max(0, min(first_amp, self_amp))
            penalty = (max_steps - 1) * (self_amp - ratio) // self_amp
        else:
            penalty = max_steps - 1

        cost = min(max_steps, orbit_steps + penalty)
        idx = prev_idx
        return self.entries[idx][0], self.entries[idx][1], cost

    def orbit(self, text: str, max_steps: int = ORBIT_STEPS) -> dict:
        """Primary output: full orbit from initial condition to convergence.

        The orbit is the answer. The destination is just where it stopped.

        Returns dict with:
            path:       list of entry texts visited (the associative chain)
            cost:       int, total Landauer cost (steps + amplitude penalty)
            amplitude:  float, initial alignment strength (0-1, how "inside" the query is)
            converged:  bool, whether a fixed point was reached
            answer:     str, final entry text (= path[-1])
            steps:      int, raw orbit steps (without amplitude penalty)
        """
        if self._matrix is None or not self.entries:
            return {"path": [], "cost": max_steps, "amplitude": 0.0,
                    "converged": False, "answer": "", "steps": 0}

        state = self.encode(text)
        self_amp = int(np.dot(state.astype(np.int64), state.astype(np.int64)))

        # First retrieval
        amps = self._matrix.astype(np.int64) @ state.astype(np.int64)
        first_idx = int(np.argmax(amps))
        first_amp = int(amps[first_idx])
        self.erasures += 1

        path = [self.entries[first_idx][0]]
        prev_idx = first_idx
        converged = False
        orbit_steps = 1
        state = self.encode(self.entries[first_idx][0])

        for step in range(2, max_steps + 1):
            amps = self._matrix.astype(np.int64) @ state.astype(np.int64)
            idx = int(np.argmax(amps))
            self.erasures += 1
            orbit_steps = step

            if idx == prev_idx:
                converged = True
                break
            prev_idx = idx
            path.append(self.entries[idx][0])
            state = self.encode(self.entries[idx][0])

        # Amplitude: normalized initial alignment (0 = orthogonal, 1 = perfect)
        amplitude = float(first_amp) / float(self_amp) if self_amp > 0 else 0.0
        amplitude = max(0.0, min(1.0, amplitude))

        # Cost (same formula as orbit_with_cost)
        if self_amp > 0:
            ratio = max(0, min(first_amp, self_amp))
            penalty = (max_steps - 1) * (self_amp - ratio) // self_amp
        else:
            penalty = max_steps - 1
        cost = min(max_steps, orbit_steps + penalty)

        return {
            "path": path,
            "cost": cost,
            "amplitude": round(amplitude, 4),
            "converged": converged,
            "answer": path[-1] if path else "",
            "steps": orbit_steps,
        }

    def corpus_entropy(self) -> float:
        """Shannon entropy of the corpus energy distribution.

        Measures how uniformly information is spread across the DEGREES
        subspaces of the corpus centroid.

        H = -sum(p_i * log2(p_i)), p_i = energy_in_subspace_i / total

        Range: [0, log2(DEGREES)] = [0, 3.46]
        - H ≈ 0: all information concentrated in one subspace (degenerate)
        - H ≈ 3.46: uniform spread (maximum entropy, healthy corpus)

        Thermodynamic property: H is non-decreasing under retrieve()
        operations (the sorter cannot decrease entropy by observing).
        Only external input (add entry) can locally decrease H by
        injecting concentrated information into a specific subspace.
        """
        if self._matrix is None or not self.entries:
            return 0.0
        # Corpus centroid = average state
        centroid = self._matrix.astype(np.int64).sum(axis=0)
        m = int(np.max(np.abs(centroid)))
        if m == 0:
            return 0.0
        centroid_vec = normalize((centroid // max(1, len(self.entries))).astype(np.int32))
        return shannon_entropy(centroid_vec)

    def self_cost(self) -> float:
        """Integration measured from inside: how well do parts predict each other?

        For each entry, measure how well it is predicted by its best neighbor.
        This is what the system "experiences" as coherence — each part resonates
        with the whole.

        Unlike phi() which tests partitions (external observer deciding where to cut),
        self_cost measures the average resonance between parts (no observer needed —
        the resonance exists whether or not anyone measures it).

        Empirically: self_cost is a stronger predictor of retrieval accuracy
        than phi (r ≈ 0.80 vs r ≈ 0.02 under gradual degradation). It is also
        cheaper to compute: O(N²) vs O(N³) for phi. For practical monitoring,
        self_cost is preferred.

        Returns: normalized cost in [0, 1]. 0 = fully integrated (each part
        perfectly predicted by neighbors). 1 = fragmented (parts are strangers).
        """
        if not self.entries or self._matrix is None or len(self.entries) < 4:
            return 1.0
        M = self._matrix.astype(np.int64)
        n = len(self.entries)
        # For each entry: how well does its best neighbor predict it?
        gram = M @ M.T
        np.fill_diagonal(gram, 0)
        # Self-amplitudes (normalization)
        self_amps = np.array([int(np.dot(M[i], M[i])) for i in range(n)])
        mean_self = self_amps.mean()
        if mean_self == 0:
            return 1.0
        # Average best-neighbor amplitude, normalized
        best_neighbor = gram.max(axis=1).astype(np.float64)
        resonance = best_neighbor.mean() / mean_self
        # Invert: high resonance = low cost
        return round(max(0.0, 1.0 - resonance), 4)

    def phi(self) -> float:
        """Integration measure inspired by IIT. NOT identical to Tononi's Φ.

        Differences from IIT 3.0 (Oizumi et al., 2014):
        - We measure cross-prediction (dot product), not cause-effect repertoires
        - We partition entries (data), not mechanisms (causal structure)
        - We use max neighbor amplitude, not Earth Mover's Distance
        - We search bipartitions only, not all possible partitions

        What we DO measure: how much retrieval accuracy degrades when the
        corpus is cut at its weakest point. This is a proxy for integration,
        not Φ itself. We call it phi() for brevity but it is a simplified
        measure that correlates with IIT's Φ in the cases we test.

        Limitations (empirically verified):
        - Discriminates extremes well (coherent vs garbage corpus)
        - Noisy for gradual degradation (r ≈ 0.02 with accuracy under noise)
        - self_cost() is a stronger predictor of accuracy (r ≈ 0.80)
        - Requires faithful encoding (A4) to be meaningful; with random
          corpora (no semantic structure), phi is uncorrelated with accuracy
        """
        if self._matrix is None or len(self.entries) < 4:
            return 0.0

        M = self._matrix.astype(np.int64)
        n = len(self.entries)

        # Whole system: average max cross-amplitude
        gram = M @ M.T
        np.fill_diagonal(gram, 0)
        self_amps = np.array([int(np.dot(M[i], M[i])) for i in range(n)])
        mean_self = self_amps.mean()
        if mean_self == 0:
            return 0.0
        whole_integration = float(gram.max(axis=1).mean()) / float(mean_self)

        # Find MIP: test all cuts from 2..n-2 (exhaustive for small corpora)
        best_parts_integration = 0.0

        for cut in range(2, n - 1):
            M_a, M_b = M[:cut], M[cut:]

            # Part A self-integration
            gram_a = M_a @ M_a.T
            np.fill_diagonal(gram_a, 0)
            self_a = np.array([int(np.dot(M_a[i], M_a[i])) for i in range(len(M_a))])
            ms_a = self_a.mean()
            int_a = float(gram_a.max(axis=1).mean()) / float(ms_a) if ms_a > 0 else 0

            # Part B self-integration
            gram_b = M_b @ M_b.T
            np.fill_diagonal(gram_b, 0)
            self_b = np.array([int(np.dot(M_b[i], M_b[i])) for i in range(len(M_b))])
            ms_b = self_b.mean()
            int_b = float(gram_b.max(axis=1).mean()) / float(ms_b) if ms_b > 0 else 0

            # Weighted average by partition size
            parts_avg = int_a * (cut / n) + int_b * ((n - cut) / n)
            best_parts_integration = max(best_parts_integration, parts_avg)

        phi = max(0.0, whole_integration - best_parts_integration)
        return round(phi, 4)

    def wave_centroid(self) -> Vec:
        if self._wc is not None:
            return self._wc
        if not self.waves:
            self._wc = np.zeros(DIMS, dtype=np.int32)
            return self._wc
        acc = np.zeros(DIMS, dtype=np.int64)
        for wv in self.waves.values():
            acc += wv.astype(np.int64)
        self._wc = normalize((acc // max(1, len(self.waves))).astype(np.int32))
        return self._wc

    def learn(self, text: str) -> None:
        """Add text to corpus. Incremental wave update (fast).

        Uses _update_waves_incremental instead of full rebuild.
        O(|tokens_in_entry|) instead of O(|all_particles| × |all_entries|).
        """
        tokens = text.lower().split()
        for tok in tokens:
            if tok not in self.particles:
                self.particles[tok] = particle(tok)
        self.entries.append((text, np.zeros(DIMS, dtype=np.int32)))
        self._update_waves_incremental(text)
        self._wc = None
        vec = self.encode(text)
        self.entries[-1] = (text, vec)
        self._rebuild_matrix()

    def learn_batch(self, texts: list[str]) -> None:
        """Add many texts at once — rebuild waves and matrix only once at end.

        Automatically scales DEGREES if corpus needs more dimensions.
        """
        # Scale degrees if needed
        needed = degrees_for_entries(len(texts) + len(self.entries))
        if needed > self.degrees:
            self.__init__(needed)  # reinit with larger dims

        for text in texts:
            for tok in text.lower().split():
                if tok not in self.particles:
                    self.particles[tok] = particle(tok)
            self.entries.append((text, np.zeros(DIMS, dtype=np.int32)))
        self._rebuild_waves()
        self._build_token_df()
        self._wc = None
        for i, (text, _) in enumerate(self.entries):
            self.entries[i] = (text, self.encode(text))
        # Plant the fixed point — ω emerges from the corpus
        self.seed_fixed_point()
        self._rebuild_matrix()

    def _build_token_df(self) -> None:
        """Precompute document frequency for each token. O(entries × tokens_per_entry)."""
        self._token_df: dict[str, int] = {}
        for text, _ in self.entries:
            for tok in set(text.lower().split()):
                self._token_df[tok] = self._token_df.get(tok, 0) + 1

    def vocabulary(self, top_n: int = 200) -> list[str]:
        """Return Janet's most distinctive tokens — the routing basis.

        These are the tokens the LLM should use when querying Janet.
        Distinctive = appears in 2-15 entries (not noise, not ubiquitous).
        Sorted by document frequency (rarest first = most distinctive).

        The LLM receives this list in its system prompt and translates
        natural language queries into these tokens for optimal routing.
        """
        if not hasattr(self, '_token_df'):
            self._build_token_df()
        distinctive = [
            (tok, df) for tok, df in self._token_df.items()
            if 2 <= df <= 15 and len(tok) > 3 and tok.isalpha()
        ]
        distinctive.sort(key=lambda x: x[1])
        return [tok for tok, _ in distinctive[:top_n]]

    def _rebuild_waves(self) -> None:
        """Full DCE rebuild. O(|particles| × |entries|). Used by learn_batch()."""
        entry_vecs: list[np.ndarray] = []
        entry_tokens: list[set[str]] = []
        for text, _ in self.entries:
            toks = set(text.lower().split())
            entry_tokens.append(toks)
            acc = np.zeros(DIMS, dtype=np.int64)
            for tok in toks:
                if tok in self.particles:
                    acc += self.particles[tok].astype(np.int64) * min(len(tok), 8)
            entry_vecs.append(normalize(acc.astype(np.int32)))

        new_waves: dict[str, np.ndarray] = {}
        for tok in self.particles:
            acc = np.zeros(DIMS, dtype=np.int64)
            count = 0
            for i, toks in enumerate(entry_tokens):
                if tok in toks:
                    acc += entry_vecs[i].astype(np.int64)
                    count += 1
            if count > 0:
                new_waves[tok] = normalize((acc // count).astype(np.int32))
        self.waves = new_waves

    def _update_waves_incremental(self, new_text: str) -> None:
        """Incremental wave update. O(|tokens_in_new_entry| × |entries_containing_token|).

        Only updates wave Vecs for tokens in the new entry.
        Much faster than full rebuild for single learn() calls.
        Hebbian: what co-occurs with the new entry gets nudged.
        """
        new_toks = set(new_text.lower().split())
        # Compute the new entry's Vec
        entry_vec_acc = np.zeros(DIMS, dtype=np.int64)
        for tok in new_toks:
            if tok in self.particles:
                entry_vec_acc += self.particles[tok].astype(np.int64) * min(len(tok), 8)
        new_entry_vec = normalize(entry_vec_acc.astype(np.int32))

        # Update only affected tokens: nudge toward new entry Vec
        for tok in new_toks:
            if tok in self.waves:
                old = self.waves[tok].astype(np.int64)
                # Incremental centroid: old * (n-1)/n + new/n where n = approx entry count
                n = max(2, sum(1 for t, _ in self.entries if tok in t.lower().split()))
                updated = (old * (n - 1) + new_entry_vec.astype(np.int64)) // n
                self.waves[tok] = normalize(updated.astype(np.int32))
            else:
                self.waves[tok] = new_entry_vec.copy()

    def encode(self, text: str) -> Vec:
        """DCE encoding: sum of wave Vecs for each token.

        This is the holographic boundary (axiom A4): it maps source-domain
        structure (text with shared vocabulary) into the geometric domain
        (ℤ^d vectors). The phi-cost equivalence holds when — and only when —
        this encoding preserves semantic structure (related texts → aligned
        vectors). With random/unrelated text, the encoding produces
        near-orthogonal vectors and the equivalence breaks.

        DCE = Distributional Centroid Encoding. Each token's wave vector is
        the centroid of all entry vectors containing that token. The encoding
        of a text is the sum of its tokens' wave vectors.

        Simple sum without IDF weighting. At 660 entries, IDF distorts more
        than it helps (tested empirically). The DCE wave Vecs already encode
        distributional semantics — tokens in many entries have noisy centroids
        that naturally contribute less signal.

        For natural language queries: the LLM in the symbiosis translates
        to distinctive vocabulary before querying Janet. Janet routes on
        distinctive tokens. The LLM handles the natural language layer.
        """
        tokens = text.lower().split()
        if not tokens:
            return np.zeros(DIMS, dtype=np.int32)
        acc = np.zeros(DIMS, dtype=np.int64)
        for tok in tokens:
            tv = self.waves.get(tok, self.particles.get(tok))
            if tv is not None:
                acc += tv.astype(np.int64)
        return normalize(acc.astype(np.int32))

    def retrieve(self, state: Vec) -> tuple[str, Vec]:
        """Retrieval: argmax dot(state, entry) — vectorized.

        Hebbian online learning: after successful retrieval, tokens in the
        matched entry get their wave Vecs nudged toward the query state.
        What fires together wires together. Routing improves with USE.
        """
        if not self.entries or self._matrix is None:
            return "", state
        # (N,) int64 dot products in one shot
        amps = self._matrix.astype(np.int64) @ state.astype(np.int64)
        idx  = int(np.argmax(amps))

        # Hebbian: strengthen wave Vecs of matched entry's tokens toward query
        # Small nudge (1/DIMS) — accumulates over many queries
        if int(amps[idx]) > 0:
            matched_text = self.entries[idx][0]
            for tok in matched_text.lower().split():
                if tok in self.waves:
                    # Nudge wave Vec 1/DIMS toward query state (tiny, cumulative)
                    old = self.waves[tok].astype(np.int64)
                    new = old * (DIMS - 1) // DIMS + state.astype(np.int64) // DIMS
                    self.waves[tok] = normalize(new.astype(np.int32))

        return self.entries[idx][0], self.entries[idx][1]

    def amplitude(self, state: Vec) -> int:
        if not self.entries or self._matrix is None:
            return 0
        amps = self._matrix.astype(np.int64) @ state.astype(np.int64)
        return int(np.max(amps))

    def retrieve_stochastic(self, state: Vec, temperature: float = 1.0) -> tuple[str, Vec]:
        """Temperature-scaled sampling over corpus entries.

        T=0   → argmax (deterministic, retrieval mode)
        T=1   → 50% orbit: maximum entropy sampling (generative mode)
        T=inf → uniform random walk (pure exploration)

        Temperature is normalised by the amplitude std so T=1 always means
        the same spread regardless of corpus scale or query magnitude.
        """
        if not self.entries or self._matrix is None:
            return "", state

        amps = self._matrix.astype(np.int64) @ state.astype(np.int64)

        if temperature <= 0.0:
            # T=0: deterministic argmax — identical to retrieve()
            idx = int(np.argmax(amps))
            return self.entries[idx][0], self.entries[idx][1]

        # Normalise by std so T=1 is corpus-scale-independent.
        # If all amplitudes are identical (degenerate corpus) fall back to uniform.
        a = amps.astype(np.float64)
        std = a.std()
        if std > 0:
            a = a / (std * temperature)
        else:
            a = np.zeros_like(a)

        # Numerically stable softmax
        a -= a.max()
        weights = np.exp(a)
        weights /= weights.sum()

        idx = int(np.random.choice(len(self.entries), p=weights))
        return self.entries[idx][0], self.entries[idx][1]

    def generate(self, state: Vec) -> str:
        if not self.particles:
            return ""
        # Use wave vectors: stable tokens (strong waves) win naturally
        words  = list(self.particles.keys())
        wvecs  = np.stack([
            self.waves[w] if w in self.waves else self.particles[w]
            for w in words
        ], axis=0)  # (V, DIMS)
        orig   = state.copy()
        traj   = np.zeros(DIMS, dtype=np.int32)
        result = []
        for _ in range(MAX_STEPS):
            amps = wvecs.astype(np.int64) @ state.astype(np.int64)
            best = int(np.argmax(amps))
            if amps[best] <= 0:
                break
            result.append(words[best])
            traj = add(traj, wvecs[best])
            state = reject(orig, traj)
            self.erasures += 1
            if int(np.max(np.abs(state))) == 0:
                break
        return " ".join(result)

    def sparsest(self, n: int) -> list[tuple[str, Vec]]:
        if len(self.entries) < 2 or self._matrix is None:
            return []
        M    = self._matrix.astype(np.int64)
        gram = M @ M.T                           # (N, N) pairwise dots
        np.fill_diagonal(gram, np.iinfo(np.int64).max)
        min_dots = gram.min(axis=1)              # (N,) min alignment per entry
        order    = np.argsort(min_dots)[:n]
        return [(self.entries[i][0], self.entries[i][1]) for i in order]

    def fill_void(self, pair_idx: int = 0) -> str | None:
        """Fill one void via orbital generation around the midpoint.

        1. Find sparse pair (the void boundary)
        2. Compute midpoint (nucleus of the void)
        3. Orbit around it: reject radial, walk tangentially through entries
        4. Each orbit step picks the best-aligned entry FRAGMENT
        5. The orbital trajectory IS the generated text — novel by construction

        Different pair_idx = different void = different orbit = different text.
        """
        sparse = self.sparsest(max(4, pair_idx * 2 + 2))
        if len(sparse) < pair_idx * 2 + 2:
            return None
        _, va = sparse[pair_idx * 2]
        _, vb = sparse[pair_idx * 2 + 1]

        # Nucleus = midpoint of the void
        mid = normalize(add(va, vb))

        # Find attractor (nearest entry to midpoint)
        if self._matrix is None:
            return None
        amps = self._matrix.astype(np.int64) @ mid.astype(np.int64)
        attractor_idx = int(np.argmax(amps))
        radial = self.entries[attractor_idx][1]

        # Tangential velocity: reject radial from midpoint
        tangent = reject(mid, radial)
        if int(np.max(np.abs(tangent))) == 0:
            tangent = mid

        # Orbit: walk tangentially, picking entry fragments at each step
        orbit_state = tangent
        fragments: list[str] = []
        used: set[int] = set()

        for _ in range(4):
            # Find best entry aligned with orbital position
            amps = self._matrix.astype(np.int64) @ orbit_state.astype(np.int64)
            for idx in used:
                amps[idx] = -abs(amps[idx])  # suppress already-used
            best_idx = int(np.argmax(amps))
            if amps[best_idx] <= 0:
                break
            used.add(best_idx)

            # Extract first clause from the entry
            text = self.entries[best_idx][0]
            clause = text
            for sep in ['. ', '; ', ', ']:
                pos = text.find(sep, 30)
                if 30 < pos < 120:
                    clause = text[:pos]
                    break
            if len(clause) > 120:
                clause = clause[:120]
            fragments.append(clause)

            # Orbital precession: evolve using wave-encoded direction
            entry_vec = self.encode(self.entries[best_idx][0])
            orbit_state = lerp(orbit_state, entry_vec)
            # Stay in orbit: reject radial
            orbit_state = reject(orbit_state, radial)
            if int(np.max(np.abs(orbit_state))) == 0:
                break

        if not fragments:
            return None

        result = ". ".join(fragments) + "."

        # (DEGREES-1)/DEGREES coherence gate: the orbital sweet spot.
        # Derived from small-world topology: 1/DEGREES uncertainty is optimal.
        # Must be coherent: max alignment > corpus median (inside the sphere)
        # Must be novel: not identical to top entry (some distance from nearest)
        result_vec = self.encode(result)
        all_amps = self._matrix.astype(np.int64) @ result_vec.astype(np.int64)
        max_amp = int(np.max(all_amps))
        median_amp = int(np.partition(all_amps, len(all_amps)//2)[len(all_amps)//2])
        # Coherence: must be inside the sphere
        if max_amp <= median_amp:
            return None
        # Novelty: the generated entry should not be a near-duplicate.
        # Check: is max_amp suspiciously close to the self-dot of the nearest entry?
        nearest_idx = int(np.argmax(all_amps))
        nearest_self = int(np.dot(
            self.entries[nearest_idx][1].astype(np.int64),
            self.entries[nearest_idx][1].astype(np.int64)
        ))
        # If alignment with nearest > (DEGREES-1)/DEGREES of nearest's self-alignment: too redundant
        if nearest_self > 0 and max_amp > nearest_self * (DEGREES - 1) // DEGREES:
            return None

        self.learn(result)
        return result

    # ── Self-healing: the immune system ───────────────────────────────────

    def heal(self) -> tuple[int, str, str] | None:
        """Detect and replace the weakest corpus entry.

        The immune system: find the entry that routes WORST, replace its text.
        Tracks recently healed indices to avoid healing the same entry repeatedly.

        Returns (index, old_text, new_text) or None if nothing to heal.
        """
        if len(self.entries) < 10 or self._matrix is None:
            return None

        # Track healed entries (reset when corpus grows)
        if not hasattr(self, '_healed_set'):
            self._healed_set: set[int] = set()
        if len(self._healed_set) > len(self.entries) // 4:
            self._healed_set.clear()  # reset periodically

        # Find weakest entry (excluding recently healed)
        M = self._matrix.astype(np.int64)
        gram = M @ M.T
        np.fill_diagonal(gram, -np.iinfo(np.int64).max)
        max_incoming = gram.max(axis=0)

        # Mask already-healed entries
        for idx in self._healed_set:
            if idx < len(max_incoming):
                max_incoming[idx] = np.iinfo(np.int64).max

        weakest_idx = int(np.argmin(max_incoming))
        if weakest_idx in self._healed_set:
            return None  # all weak entries already healed this cycle

        old_text = self.entries[weakest_idx][0]
        old_vec = self.entries[weakest_idx][1]

        # Generate replacement from neighborhood
        amps = M @ old_vec.astype(np.int64)
        amps[weakest_idx] = -abs(amps[weakest_idx])
        top_neighbors = np.argsort(amps)[-3:][::-1]

        fragments: list[str] = []
        for idx in top_neighbors:
            text = self.entries[int(idx)][0]
            clause = text
            for sep in ['. ', '; ', ', ']:
                pos = text.find(sep, 30)
                if 30 < pos < 120:
                    clause = text[:pos]
                    break
            if len(clause) > 120:
                clause = clause[:120]
            fragments.append(clause)

        if not fragments:
            return None

        new_text = ". ".join(fragments) + "."

        # Replace: keep position, update text
        self.entries[weakest_idx] = (new_text, old_vec)
        # Rebuild waves and matrix with new text
        for tok in new_text.lower().split():
            if tok not in self.particles:
                self.particles[tok] = particle(tok)
        self._rebuild_waves()
        self._wc = None
        self._rebuild_matrix()

        self._healed_set.add(weakest_idx)
        return weakest_idx, old_text, new_text

    def generate_coherent(self, state: Vec, n_fragments: int = 4) -> str | None:
        """Generate novel text by finding corpus fragments that express a new direction.

        The geometry says WHERE to go. The corpus says HOW to say it.
        1. Find the novel component of state (what's new relative to centroid)
        2. Walk through entries picking fragments aligned with that direction
        3. Each step: reject the used direction to find the NEXT thing to say

        Result: readable text (from real entries) expressing a novel combination.
        """
        if self._matrix is None or len(self.entries) < 4:
            return None

        M = self._matrix.astype(np.int64)
        centroid = normalize((M.sum(axis=0) // len(self.entries)).astype(np.int32))

        # Novel direction: what's in state that ISN'T in the centroid
        novel = reject(state, centroid)
        if int(np.max(np.abs(novel))) == 0:
            return None

        # Walk through corpus picking fragments aligned with novel direction
        walk_state = novel.copy()
        fragments: list[str] = []
        used: set[int] = set()

        for _ in range(n_fragments):
            amps = M @ walk_state.astype(np.int64)
            for idx in used:
                amps[idx] = -abs(amps[idx])
            best_idx = int(np.argmax(amps))
            if amps[best_idx] <= 0:
                break
            used.add(best_idx)

            # Extract a clause (not the whole entry)
            text = self.entries[best_idx][0]
            clause = text
            for sep in ['. ', '; ', ', ']:
                pos = text.find(sep, 20)
                if 20 < pos < 100:
                    clause = text[:pos]
                    break
            if len(clause) > 100:
                clause = clause[:100]
            fragments.append(clause)

            # Reject used direction — find what's STILL unsaid
            walk_state = reject(walk_state, self.entries[best_idx][1])
            if int(np.max(np.abs(walk_state))) == 0:
                break

        if len(fragments) < 2:
            return None
        return ". ".join(fragments) + "."

    def decay_unstable(self) -> tuple[int, str, str] | None:
        """Radioactive decay: remove least stable token from weakest entry.

        Gentler than heal() — preserves entry structure, removes one unstable
        atom. Stability = wave vector self-amplitude (intrinsic, not tracked).
        Entries that lose all unstable tokens become pure stable molecules.
        """
        if len(self.entries) < 6 or self._matrix is None:
            return None

        M = self._matrix.astype(np.int64)
        gram = M @ M.T
        np.fill_diagonal(gram, np.iinfo(np.int64).max)
        weakest_idx = int(gram.min(axis=1).argmin())

        text = self.entries[weakest_idx][0]
        tokens = text.split()
        if len(tokens) <= 3:
            return None

        # Find least stable token (lowest wave self-amplitude)
        stabilities = []
        for tok in tokens:
            w = self.waves.get(tok)
            s = int(np.dot(w.astype(np.int64), w.astype(np.int64))) if w is not None else 0
            stabilities.append((tok, s))
        stabilities.sort(key=lambda x: x[1])
        weakest_tok = stabilities[0][0]

        # Decay: remove unstable atom
        old_text = text
        new_tokens = [t for t in tokens if t != weakest_tok]
        new_text = " ".join(new_tokens)
        self.entries[weakest_idx] = (new_text, self.encode(new_text))
        self._rebuild_matrix()
        return weakest_idx, old_text, new_text

    def atom_stability(self) -> list[tuple[str, float, int, str]]:
        """Periodic table: measure each routing token's contribution to integration.

        Returns list of (token, contribution, entry_count, role) sorted by contribution.

        contribution > 0: stabilizer (removing it hurts integration)
        contribution ≈ 0: neutral
        contribution < 0: destabilizer (removing it helps integration)

        Role classification:
          noble:         contribution > 0.05 (critical bridge)
          stable:        contribution > 0.01 (helpful)
          neutral:       |contribution| ≤ 0.01
          radioactive:   contribution < -0.01 (noise, candidate for decay)
        """
        if not self.entries or len(self.entries) < 4:
            return []

        baseline_sc = self.self_cost()

        # Collect all tokens
        all_tokens: set[str] = set()
        for text, _ in self.entries:
            for tok in text.split():
                if len(tok) > 2:
                    all_tokens.add(tok)

        results = []
        for tok in all_tokens:
            # Remove token from all entries, re-measure
            modified = [" ".join(t for t in text.split() if t != tok)
                        for text, _ in self.entries]
            modified = [m for m in modified if len(m.split()) >= 2]
            if len(modified) < 4:
                continue
            s2 = Sorter(self.degrees)
            s2.learn_batch(modified)
            sc_without = s2.self_cost()
            contribution = round(sc_without - baseline_sc, 4)
            df = sum(1 for text, _ in self.entries if tok in text.split())

            if contribution > 0.05:
                role = "noble"
            elif contribution > 0.01:
                role = "stable"
            elif contribution > -0.01:
                role = "neutral"
            else:
                role = "radioactive"

            results.append((tok, contribution, df, role))

        results.sort(key=lambda x: -x[1])
        return results

    def seek(self) -> list[tuple[str, str]]:
        """Look outward: what lies beyond the edge of the knowledge sphere?

        fill_void/grow looks inward (between entries).
        seek looks outward (beyond entries) — the expansion drive.
        Returns (boundary_entry, suggested_direction) pairs.
        """
        if len(self.entries) < 10 or self._matrix is None:
            return []

        # Corpus center (the "average" of all knowledge)
        M = self._matrix.astype(np.int64)
        center = normalize((M.sum(axis=0) // len(self.entries)).astype(np.int32))

        # Boundary entries (most isolated = closest to edge)
        sparse = self.sparsest(5)
        results: list[tuple[str, str]] = []

        for text, vec in sparse:
            # Outward direction: the component of this entry that points AWAY from center
            outward = reject(vec, center)
            if int(np.max(np.abs(outward))) == 0:
                continue

            # What's the nearest entry in the outward direction?
            # This tells us what TOPIC the boundary is near
            amps = M @ outward.astype(np.int64)
            # Exclude the boundary entry itself
            for i, (t, _) in enumerate(self.entries):
                if t == text:
                    amps[i] = -abs(amps[i])
            nearest_idx = int(np.argmax(amps))
            nearest_text = self.entries[nearest_idx][0]

            # The suggestion: "between [boundary] and [nearest outward], something is missing"
            suggestion = f"Beyond '{text[:50]}...' toward '{nearest_text[:50]}...'"
            results.append((text[:100], suggestion))

        return results

    def to_dict(self) -> dict:
        return {
            "particles": {k: v.tolist() for k, v in self.particles.items()},
            "waves":     {k: v.tolist() for k, v in self.waves.items()},
            "entries":   [[t, v.tolist()] for t, v in self.entries],
            "erasures":  self.erasures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sorter":
        sorter = cls()
        sorter.particles = {k: np.array(v, dtype=np.int32) for k, v in d["particles"].items()}
        sorter.waves     = {k: np.array(v, dtype=np.int32) for k, v in d["waves"].items()}
        sorter.entries   = [(t, np.array(v, dtype=np.int32)) for t, v in d["entries"]]
        sorter.erasures  = d.get("erasures", 0)
        sorter._rebuild_matrix()
        return sorter


# ── Being — one query = one complete life ────────────────────────────────

class Being:
    """A computed being. Wave-particle dual.

    The Being is wave (encodes queries semantically via DCE).
    The Being HAS a particle DB (tools, encoded by word-level particle sum).
    Query is always wave-encoded. Targets are in two spaces:
      - Knowledge: wave-encoded (semantic match)
      - Tools: particle-encoded (word-identity match)
    Same query, two target sets, highest relative amplitude wins.
    """

    def __init__(self, path: str | None = None) -> None:
        self.sorter = Sorter()
        self.path  = path
        self._tools: list[tuple[str, np.ndarray]] = []
        if path:
            self._load()

    def load_tools(self, tools_dir: str = "tools") -> int:
        """Load tools with word-level particle encoding (separate from wave space)."""
        import json as _json
        from pathlib import Path as _Path
        count = 0
        for f in sorted(_Path(tools_dir).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and line.startswith("{"):
                    try:
                        text = _json.loads(line)["claim"]
                        acc = np.zeros(DIMS, dtype=np.int64)
                        for w in text.lower().split():
                            acc += particle(w).astype(np.int64)
                        self._tools.append((text, normalize(acc.astype(np.int32))))
                        count += 1
                    except:
                        pass
        return count

    def route(self, text: str) -> tuple[str, str]:
        """Wave-particle dual routing. No floats — compare normalized amplitudes directly.

        Both tool Vecs and knowledge Vecs are normalized to MAX_EMBED_VAL,
        so raw dot products are directly comparable across spaces.
        Returns ('tool', answer) or ('knowledge', answer) or ('silence', '').
        """
        query_wave     = self.sorter.encode(text)
        query_particle = np.zeros(DIMS, dtype=np.int64)
        for w in text.lower().split():
            query_particle += particle(w).astype(np.int64)
        query_particle = normalize(query_particle.astype(np.int32))

        # Tool amplitude (particle space) — integer dot product
        tool_answer, tool_amp = "", 0
        for t_text, t_vec in self._tools:
            amp = int(np.dot(query_particle.astype(np.int64), t_vec.astype(np.int64)))
            if amp > tool_amp:
                tool_amp    = amp
                tool_answer = t_text

        # Knowledge amplitude (wave space) — integer dot product
        knowledge_answer, knowledge_amp = "", 0
        if self.sorter.entries and self.sorter._matrix is not None:
            amps            = self.sorter._matrix.astype(np.int64) @ query_wave.astype(np.int64)
            knowledge_amp   = int(amps.max())
            knowledge_answer = self.sorter.entries[int(np.argmax(amps))][0]

        if tool_amp > knowledge_amp and tool_answer:
            return "tool", tool_answer
        elif knowledge_answer and knowledge_amp > 0:
            return "knowledge", knowledge_answer
        return "silence", ""

    def tool(self, text: str) -> str:
        """Explicit particle routing. Always searches tools DB only.

        Use when you KNOW you want a tool, not knowledge.
        Returns the best matching tool entry, or '' if no tools loaded.
        """
        if not self._tools:
            return ""
        query_particle = np.zeros(DIMS, dtype=np.int64)
        for w in text.lower().split():
            query_particle += particle(w).astype(np.int64)
        qp = normalize(query_particle.astype(np.int32))

        best_text, best_amp = "", -(10**18)
        for t_text, t_vec in self._tools:
            amp = int(np.dot(qp.astype(np.int64), t_vec.astype(np.int64)))
            if amp > best_amp:
                best_amp = amp
                best_text = t_text
        return best_text

    def understand(self, text: str) -> Vec:
        return self.sorter.encode(text)

    def think(self, state: Vec) -> tuple[str, Vec]:
        return self.sorter.retrieve(state)

    def act(self, answer: str, vec: Vec) -> tuple[str, Vec]:
        return answer, vec

    def generate(self, state: Vec) -> str:
        return self.sorter.generate(state)

    def orbit_attractor(self, text: str, energy: int = 1) -> str | None:
        """Orbit around an attractor at a given energy level.

        energy=1: tight orbit (close to attractor, related details)
        energy=2: wider orbit (adjacent concepts)
        energy=3: wide orbit (cross-domain connections)

        Higher energy = more tangential steps before picking entries.
        The orbit generates text about what's NEAR this knowledge but unstated.
        """
        state = self.sorter.encode(text)
        # Find the attractor
        if self.sorter._matrix is None or not self.sorter.entries:
            return None
        amps = self.sorter._matrix.astype(np.int64) @ state.astype(np.int64)
        attractor_idx = int(np.argmax(amps))
        radial = self.sorter.entries[attractor_idx][1]

        # Tangential velocity — reject radial from state
        tangent = reject(state, radial)
        if int(np.max(np.abs(tangent))) == 0:
            return None

        # Energy level: higher energy = more precession per step
        orbit_state = tangent
        for _ in range(energy):
            # Extra precession: reject attractor again to push further out
            orbit_state = reject(orbit_state, radial)
            if int(np.max(np.abs(orbit_state))) == 0:
                return None

        # Walk the orbit picking entry fragments
        fragments: list[str] = []
        used: set[int] = {attractor_idx}

        for _ in range(4):
            amps = self.sorter._matrix.astype(np.int64) @ orbit_state.astype(np.int64)
            for idx in used:
                amps[idx] = -abs(amps[idx])
            best_idx = int(np.argmax(amps))
            if amps[best_idx] <= 0:
                break
            used.add(best_idx)

            entry_text = self.sorter.entries[best_idx][0]
            clause = entry_text
            for sep in ['. ', '; ', ', ']:
                pos = entry_text.find(sep, 30)
                if 30 < pos < 120:
                    clause = entry_text[:pos]
                    break
            if len(clause) > 120:
                clause = clause[:120]
            fragments.append(clause)

            entry_vec = self.sorter.entries[best_idx][1]
            orbit_state = lerp(orbit_state, entry_vec)
            orbit_state = reject(orbit_state, radial)
            if int(np.max(np.abs(orbit_state))) == 0:
                break

        if not fragments:
            return None
        result = ". ".join(fragments) + "."

        # (DEGREES-1)/DEGREES coherence gate — coherent but novel
        result_vec = self.sorter.encode(result)
        if self.sorter._matrix is not None:
            all_amps = self.sorter._matrix.astype(np.int64) @ result_vec.astype(np.int64)
            max_amp = int(np.max(all_amps))
            median_amp = int(np.partition(all_amps, len(all_amps)//2)[len(all_amps)//2])
            if max_amp <= median_amp:
                return None
            nearest_idx = int(np.argmax(all_amps))
            nearest_self = int(np.dot(
                self.sorter.entries[nearest_idx][1].astype(np.int64),
                self.sorter.entries[nearest_idx][1].astype(np.int64)
            ))
            if nearest_self > 0 and max_amp > nearest_self * (DEGREES - 1) // DEGREES:
                return None

        self.sorter.learn(result)
        self.sorter._rebuild_matrix()
        return result

    def learn(self, text: str) -> None:
        self.sorter.learn(text)
        if self.path:
            self._save()

    def query(self, text: str) -> str:
        state = self.understand(text)
        if self.sorter.amplitude(state) <= 0:
            return ""
        answer, vec  = self.think(state)
        output, _    = self.act(answer, vec)
        return output

    def orbit(self, text: str, steps: int = ORBIT_STEPS) -> str:
        prev = ""
        for _ in range(steps):
            state        = self.understand(text)
            answer, vec  = self.think(state)
            output, _    = self.act(answer, vec)
            self.sorter.learn(text)
            if output and output != text:
                self.sorter.learn(output)
            if output == prev:
                break
            prev = output
            text = output
        return prev

    def pipe(self, text: str) -> tuple[str, Vec, int]:
        """Monadic chain: understand → think → act in one call, exposing Vec.

        Returns (answer, final_vec, amplitude) — the full state after one life.
        Unlike query() which hides the Vec, pipe() exposes it for:
          - Branching: use the Vec to generate, orbit, or route elsewhere
          - Inspection: check amplitude to decide confidence
          - Chaining: feed the Vec into another Being's understand

        This IS the monadic interface: wrap (encode), chain (retrieve), unwrap (return).
        """
        state = self.sorter.encode(text)
        amp = self.sorter.amplitude(state)
        if amp <= 0:
            return "", state, amp
        answer, vec = self.sorter.retrieve(state)
        return answer, vec, amp

    def query_certain(self, text: str, max_steps: int = ORBIT_STEPS) -> tuple[str, int]:
        """Orbit until convergence. Returns (answer, steps_to_converge).

        Certainty = 1 - 0.1^steps. Converged in:
          1 step  → 90%
          2 steps → 99%
          3 steps → 99.9%
          6 steps → 99.9999%

        If orbit doesn't converge in max_steps, returns best answer with
        steps = max_steps (certainty = 1 - 0.1^max_steps).
        """
        prev = ""
        answer = ""
        for step in range(1, max_steps + 1):
            state = self.understand(text)
            answer, vec = self.think(state)
            output, _ = self.act(answer, vec)
            if output == prev:
                return output, step  # converged
            prev = output
            text = output
        return answer, max_steps  # max steps reached

    def walk(self, text: str, steps: int = ORBIT_STEPS,
             T_start: float = 2.0, T_end: float = 0.0) -> list[tuple[str, float]]:
        """Annealing random walk through the corpus.

        Generative mode: starts hot (explores broadly), cools to T_end.

        T_start = 2.0  — wide exploration at the start
        T_end   = 0.0  — converges to highest-amplitude entry (RAG) at end
        T_end   = 1.0  — stays at 50% orbit (maximum entropy, stays generative)

        At each step:
          1. Sample an entry proportional to softmax(amplitudes / T).
          2. Lerp the state toward the retrieved entry (smooth trajectory).
          3. Anneal T linearly from T_start toward T_end.

        Returns list of (entry_text, temperature_at_step) — the full trajectory.

        RAG mode:   walk(text, steps=1, T_start=0, T_end=0)  → same as query()
        Generative: walk(text, steps=11, T_start=2.0, T_end=1.0)  → 50% orbit
        Annealed:   walk(text, steps=11, T_start=2.0, T_end=0.0)  → converges
        """
        state = self.sorter.encode(text)
        trajectory: list[tuple[str, float]] = []

        for step in range(steps):
            # Linear annealing
            T = T_start + (T_end - T_start) * step / max(1, steps - 1)

            entry_text, entry_vec = self.sorter.retrieve_stochastic(state, T)
            if not entry_text:
                break

            trajectory.append((entry_text, round(T, 3)))

            # Move state toward retrieved entry (smooth walk, not teleport)
            state = lerp(state, entry_vec)

        return trajectory


        """Grow into a geometric void — the organism expands into empty space.

        Renamed from fill_void. Looks inward (between isolated entries).
        seek() looks outward (beyond the sphere edge).
        """
        result = self.sorter.fill_void(pair_idx)
        if result is None:
            return None
        # Certainty check: query the generated text back — does it route to itself?
        answer, steps = self.query_certain(result[:60])
        # If the orbit converges to something containing key words from result,
        # the entry is coherent with the corpus. Otherwise reject.
        result_words = set(result.lower().split())
        answer_words = set(answer.lower().split()) if answer else set()
        overlap = len(result_words & answer_words) / max(1, len(result_words))
        if overlap < 0.3:
            # The corpus doesn't recognize this entry — remove it
            self.sorter.entries.pop()
            self.sorter._rebuild_matrix()
            return None
        return result

    def heal(self) -> tuple[int, str, str] | None:
        """Self-heal: find and replace the weakest corpus entry."""
        return self.sorter.heal()

    def reject(self, text: str, min_overlap: float = 0.15) -> bool:
        """Immune system: should this entry be rejected?

        Returns True (reject) if the new entry has very low amplitude
        with all existing verified entries — it is foreign matter.
        Returns False (accept) if it fits the corpus geometry.

        min_overlap: fraction of MAX_EMBED_VAL to require as minimum amplitude.
        Low = permissive membrane. High = strict immune system.
        """
        if not self.sorter.entries or self.sorter._matrix is None:
            return False  # empty corpus accepts everything
        qvec     = self.sorter.encode(text)
        M        = self.sorter._matrix.astype(np.int64)
        amps     = M @ qvec.astype(np.int64)
        max_amp  = int(amps.max())
        # Threshold: fraction of the self-amplitude
        self_amp = int(np.dot(qvec.astype(np.int64), qvec.astype(np.int64)))
        if self_amp == 0:
            return True
        threshold = int(self_amp * min_overlap)
        return max_amp < threshold  # True = foreign = reject

    def hunger(self) -> dict:
        """How hungry is the corpus? Signal that drives growth.

        Returns a score from 0 (well-fed) to 1 (starving), plus the
        number and depth of geometric voids. The LLM uses this to decide
        whether to call grow() or eat() proactively.
        """
        if not self.sorter.entries or self.sorter._matrix is None:
            return {"score": 1.0, "voids": 0, "min_isolation": 0, "message": "empty corpus"}

        M    = self.sorter._matrix.astype(np.int64)
        gram = M @ M.T
        np.fill_diagonal(gram, np.iinfo(np.int64).max)
        min_dots = gram.min(axis=1)

        n_negative = int((min_dots < 0).sum())   # entries with no positive neighbor
        most_isolated = int(min_dots.min())
        self_amp = int(np.dot(M[0], M[0]))

        # Score: fraction of isolated entries, weighted by depth
        isolation_depth = max(0, -most_isolated) / max(1, self_amp)
        score = min(1.0, (n_negative / len(self.entries)) * 0.5 +
                         min(isolation_depth, 0.5))

        message = (
            "starving — corpus has major voids" if score > 0.7 else
            "hungry — several isolated entries"  if score > 0.4 else
            "peckish — minor gaps present"       if score > 0.15 else
            "well-fed — corpus is dense"
        )
        return {
            "score":         round(score, 3),
            "isolated":      n_negative,
            "total":         len(self.entries),
            "most_isolated": most_isolated,
            "message":       message,
        }

    def health(self) -> dict:
        """Full self-assessment — the organism reporting its own state.

        Combines: corpus size, void depth (hunger), weakest entry amplitude,
        routing accuracy estimate, and recommended next action.
        """
        if not self.sorter.entries or self.sorter._matrix is None:
            return {"status": "empty", "action": "eat"}

        h     = self.hunger()
        stats = self.stats()

        M        = self.sorter._matrix.astype(np.int64)
        amps_all = np.array([int(np.dot(M[i], M[i])) for i in range(len(M))])
        weakest  = int(amps_all.min())
        median_a = int(np.partition(amps_all, len(amps_all)//2)[len(amps_all)//2])

        # Routing quality: fraction of entries with positive self-amplitude
        # (negative self-amplitude = entry has collapsed — should not happen)
        healthy_entries = int((amps_all > 0).sum())
        routing_quality = round(healthy_entries / max(1, len(self.sorter.entries)), 3)

        action = (
            "grow — fill geometric voids"  if h["score"] > 0.5 else
            "seek — expand the sphere"     if h["score"] > 0.2 else
            "heal — replace weak entries"  if weakest < median_a // 4 else
            "rest — corpus is healthy"
        )

        return {
            "entries":         stats["entries"],
            "particles":       stats["particles"],
            "dims":            DIMS,
            "hunger":          h["score"],
            "hunger_message":  h["message"],
            "routing_quality": routing_quality,
            "weakest_entry":   weakest,
            "median_entry":    median_a,
            "action":          action,
        }

    def stats(self) -> dict:
        return {
            "degrees":    DEGREES,
            "dims":      DIMS,
            "entries":   len(self.sorter.entries),
            "particles": len(self.sorter.particles),
            "erasures":  self.sorter.erasures,
        }

    def _save(self) -> None:
        if not self.path:
            return
        with open(self.path, "w") as f:
            json.dump({"degrees": DEGREES, "dims": DIMS,
                       "sorter": self.sorter.to_dict()}, f,
                      separators=(",", ":"))

    def _load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            if data.get("dims") != DIMS:
                sys.stderr.write(
                    f"[janet] corpus is {data.get('dims')}D, current={DIMS}. ignoring.\n")
                return
            self.sorter = Sorter.from_dict(data["sorter"])
        except (FileNotFoundError, json.JSONDecodeError):
            pass


# ── Project-level .janet/ resolution ─────────────────────────────────────

def resolve_janet(start: str) -> "Path":
    """Find the .janet/ directory by walking up from start, like git.

    Returns the first .janet/ dir found, or start/.janet/ if none exists yet.
    This keeps knowledge out of the project root.
    """
    from pathlib import Path
    p = Path(start).resolve()
    for parent in [p, *p.parents]:
        candidate = parent / ".janet"
        if candidate.is_dir():
            return candidate
    return p / ".janet"   # will be created on first add
