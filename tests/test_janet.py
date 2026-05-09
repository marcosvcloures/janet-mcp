"""Unit tests for janet.py — quantum embedding engine."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from janet import (
    DEGREES, DIMS, MAX_EMBED_VAL, ORBIT_STEPS, MAX_STEPS, IDEAL_CORPUS,
    MIX_NUM, MIX_DEN,
    dot, add, lerp, normalize, reject, particle, Sorter, Being,
    _particle_cache,
)


# ── Fixture: small corpus ─────────────────────────────────────────────────

@pytest.fixture
def small_being():
    b = Being()
    entries = [
        "the permanent polynomial uses trivial S_n symmetry",
        "the determinant polynomial uses sign S_n symmetry",
        "GL_n column action commutes with S_n row action",
        "orbit closure preserves isotypic components",
        "P is not equal to NP via graded separation theorem",
        "Kronecker coefficients count tensor product multiplicities",
        "consciousness is integrated information Phi greater than zero",
        "cooperation is the Nash equilibrium of the infinite game",
    ]
    for e in entries:
        b.sorter.learn(e)
    return b


# ── Constants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_dims_is_power_of_two(self):
        assert DIMS == 1 << DEGREES

    def test_max_embed_val_prevents_overflow(self):
        # (MAX_EMBED_VAL)^2 * DIMS must fit int64
        assert MAX_EMBED_VAL ** 2 * DIMS < 2**63

    def test_orbit_steps_equals_degrees(self):
        assert ORBIT_STEPS == DEGREES

    def test_ideal_corpus_equals_dims(self):
        assert IDEAL_CORPUS == DIMS


# ── Integer arithmetic ────────────────────────────────────────────────────

class TestArithmetic:
    def test_dot_basic(self):
        a = np.ones(DIMS, dtype=np.int32) * 100
        b = np.ones(DIMS, dtype=np.int32) * 200
        assert dot(a, b) == 100 * 200 * DIMS

    def test_dot_orthogonal(self):
        a = np.zeros(DIMS, dtype=np.int32)
        a[0] = MAX_EMBED_VAL
        b = np.zeros(DIMS, dtype=np.int32)
        b[1] = MAX_EMBED_VAL
        assert dot(a, b) == 0

    def test_add_clips(self):
        a = np.full(DIMS, MAX_EMBED_VAL, dtype=np.int32)
        b = np.full(DIMS, MAX_EMBED_VAL, dtype=np.int32)
        result = add(a, b)
        assert int(np.max(result)) <= MAX_EMBED_VAL

    def test_normalize_scales_down(self):
        v = np.full(DIMS, MAX_EMBED_VAL * 3, dtype=np.int32)
        n = normalize(v)
        assert int(np.max(np.abs(n))) <= MAX_EMBED_VAL

    def test_normalize_preserves_small(self):
        v = np.array([1, -2, 3] + [0] * (DIMS - 3), dtype=np.int32)
        n = normalize(v)
        np.testing.assert_array_equal(v, n)

    def test_reject_orthogonality(self):
        a = np.zeros(DIMS, dtype=np.int32)
        a[0] = MAX_EMBED_VAL
        a[1] = MAX_EMBED_VAL // 2
        d = np.zeros(DIMS, dtype=np.int32)
        d[0] = MAX_EMBED_VAL
        r = reject(a, d)
        # Result should be orthogonal to d
        assert abs(dot(r, d)) < 100  # near-zero (integer rounding)

    def test_reject_zero_direction(self):
        v = np.ones(DIMS, dtype=np.int32) * 100
        d = np.zeros(DIMS, dtype=np.int32)
        r = reject(v, d)
        np.testing.assert_array_equal(r, v)

    def test_lerp_moves_toward_target(self):
        a = np.zeros(DIMS, dtype=np.int32)
        a[0] = MAX_EMBED_VAL
        b = np.zeros(DIMS, dtype=np.int32)
        b[1] = MAX_EMBED_VAL
        result = lerp(a, b)
        # Should have some component along b's direction
        assert result[1] != 0


# ── Particle embedding ────────────────────────────────────────────────────

class TestParticle:
    def test_returns_correct_shape(self):
        from janet import Sorter, DIMS as _  # ensure globals are set
        d = Sorter()  # reset to MIN_DEGREES
        from janet import DIMS
        v = particle("hello")
        assert v.shape == (DIMS,)
        assert v.dtype == np.int32

    def test_deterministic(self):
        v1 = particle("quantum")
        _particle_cache.pop("quantum", None)
        v2 = particle("quantum")
        np.testing.assert_array_equal(v1, v2)

    def test_different_words_different_vecs(self):
        v1 = particle("cat")
        v2 = particle("dog")
        assert not np.array_equal(v1, v2)

    def test_empty_string_is_zero(self):
        v = particle("")
        assert int(np.max(np.abs(v))) == 0

    def test_within_bounds(self):
        v = particle("supercalifragilistic")
        assert int(np.max(np.abs(v))) <= MAX_EMBED_VAL

    def test_caching(self):
        _particle_cache.pop("testcache", None)
        v1 = particle("testcache")
        assert "testcache" in _particle_cache
        v2 = particle("testcache")
        assert v1 is v2  # same object from cache


# ── Sorter class ───────────────────────────────────────────────────────────

class TestDemon:
    @pytest.fixture
    def sorter(self):
        d = Sorter()
        d.learn_batch([
            "photosynthesis converts sunlight into chemical energy",
            "mitochondria produce ATP through oxidative phosphorylation",
            "DNA stores genetic information as nucleotide sequences",
        ])
        return d

    def test_learn_adds_entry(self, sorter):
        assert len(sorter.entries) == 3

    def test_encode_returns_vec(self, sorter):
        v = sorter.encode("energy production")
        assert v.shape == (DIMS,)
        assert v.dtype == np.int32

    def test_retrieve_finds_relevant(self, sorter):
        state = sorter.encode("sunlight energy plants")
        text, vec = sorter.retrieve(state)
        assert "photosynthesis" in text.lower()

    def test_retrieve_empty_corpus(self):
        d = Sorter()
        text, vec = d.retrieve(np.ones(DIMS, dtype=np.int32))
        assert text == ""

    def test_amplitude_positive_for_known(self, sorter):
        state = sorter.encode("DNA genetic")
        amp = sorter.amplitude(state)
        assert amp > 0

    def test_generate_produces_words(self, sorter):
        state = sorter.encode("energy")
        result = sorter.generate(state)
        assert len(result) > 0

    def test_wave_centroid_shape(self, sorter):
        wc = sorter.wave_centroid()
        assert wc.shape == (DIMS,)

    def test_sparsest_returns_entries(self, sorter):
        sparse = sorter.sparsest(2)
        assert len(sparse) == 2
        for text, vec in sparse:
            assert isinstance(text, str)
            assert vec.shape == (DIMS,)

    def test_learn_single(self):
        d = Sorter()
        d.learn("test entry one")
        assert len(d.entries) == 1
        d.learn("test entry two")
        assert len(d.entries) == 2

    def test_vocabulary(self, sorter):
        vocab = sorter.vocabulary(top_n=10)
        assert isinstance(vocab, list)
        for tok in vocab:
            assert isinstance(tok, str)


# ── Dual operating modes ──────────────────────────────────────────────────

class TestDualModes:
    """Tests for RAG mode (T=0) and generative mode (T>0)."""

    # ── retrieve_stochastic ───────────────────────────────────────────────

    def test_rag_mode_is_deterministic(self, small_being):
        """T=0 must return the same entry every time — identical to query()."""
        query = "permanent determinant separation"
        state = small_being.sorter.encode(query)
        results = {small_being.sorter.retrieve_stochastic(state, temperature=0.0)[0]
                   for _ in range(20)}
        assert len(results) == 1, "T=0 must be deterministic"

    def test_generative_mode_is_stochastic(self, small_being):
        """T=1 must return different entries across multiple calls."""
        query = "permanent determinant separation"
        state = small_being.sorter.encode(query)
        np.random.seed(42)
        results = {small_being.sorter.retrieve_stochastic(state, temperature=1.0)[0]
                   for _ in range(30)}
        assert len(results) > 1, "T=1 must be stochastic"

    def test_stochastic_returns_valid_entry(self, small_being):
        """retrieve_stochastic must always return a corpus entry."""
        corpus_texts = {text for text, _ in small_being.sorter.entries}
        state = small_being.sorter.encode("orbit")
        for T in [0.0, 0.5, 1.0, 2.0]:
            text, vec = small_being.sorter.retrieve_stochastic(state, T)
            assert text in corpus_texts, f"T={T}: returned text not in corpus"
            assert vec.shape == (DIMS,)

    def test_high_temperature_explores_corpus(self, small_being):
        """Very high T should sample broadly — all entries reachable."""
        query = "permanent"
        state = small_being.sorter.encode(query)
        np.random.seed(0)
        seen = set()
        for _ in range(300):
            text, _ = small_being.sorter.retrieve_stochastic(state, temperature=10.0)
            seen.add(text)
        # At T=10 almost all 8 entries should be reachable
        assert len(seen) >= 6, f"High T should explore broadly; only saw {len(seen)} entries"

    def test_t0_matches_retrieve(self, small_being):
        """T=0 stochastic must match deterministic retrieve()."""
        query = "determinant sign symmetry"
        state = small_being.sorter.encode(query)
        det_text, _ = small_being.sorter.retrieve(state)
        sto_text, _ = small_being.sorter.retrieve_stochastic(state, temperature=0.0)
        assert det_text == sto_text, "T=0 must match argmax retrieve()"

    # ── walk ──────────────────────────────────────────────────────────────

    def test_walk_returns_trajectory(self, small_being):
        """walk() must return a list of (text, temperature) tuples."""
        np.random.seed(1)
        traj = small_being.walk("orbit closure permanent", steps=5)
        assert isinstance(traj, list)
        assert len(traj) == 5
        for text, T in traj:
            assert isinstance(text, str) and len(text) > 0
            assert isinstance(T, float)

    def test_walk_annealing_cools(self, small_being):
        """Temperature in walk() must decrease from T_start toward T_end."""
        np.random.seed(2)
        traj = small_being.walk("orbit", steps=6, T_start=3.0, T_end=0.0)
        temps = [T for _, T in traj]
        assert temps[0] >= temps[-1], "Temperature must cool over the walk"
        assert temps[0] == pytest.approx(3.0)
        assert temps[-1] == pytest.approx(0.0)

    def test_walk_rag_mode(self, small_being):
        """walk with T_start=T_end=0 must be deterministic (RAG mode)."""
        query = "permanent trivial symmetry"
        results = set()
        for seed in range(10):
            np.random.seed(seed)
            traj = small_being.walk(query, steps=3, T_start=0.0, T_end=0.0)
            results.add(traj[0][0])
        assert len(results) == 1, "T=0 walk must be deterministic"

    def test_walk_generative_mode_explores(self, small_being):
        """walk at T_start=T_end=1 (50% orbit) must visit multiple entries."""
        np.random.seed(42)
        seen = set()
        for _ in range(20):
            traj = small_being.walk("orbit", steps=4, T_start=1.0, T_end=1.0)
            for text, _ in traj:
                seen.add(text)
        assert len(seen) > 2, "50% orbit walk must explore the corpus"

    def test_walk_all_entries_from_corpus(self, small_being):
        """Every entry returned by walk must be in the corpus."""
        corpus_texts = {text for text, _ in small_being.sorter.entries}
        np.random.seed(7)
        traj = small_being.walk("Kronecker orbit separation", steps=8,
                                T_start=2.0, T_end=0.5)
        for text, _ in traj:
            assert text in corpus_texts

    def test_walk_state_evolves(self, small_being):
        """Annealed walk (T: 2→0) last entry should be high-amplitude."""
        query = "permanent determinant orbit closure"
        np.random.seed(0)
        traj = small_being.walk(query, steps=8, T_start=2.0, T_end=0.0)
        # Last entry (after cooling) should be high-amplitude for the query
        last_text = traj[-1][0]
        state = small_being.sorter.encode(query)
        amps = small_being.sorter._matrix.astype(np.int64) @ state.astype(np.int64)
        best_text = small_being.sorter.entries[int(np.argmax(amps))][0]
        # After annealing the walk converges near the top entry
        # (not guaranteed to be exactly the top due to lerp drift, but should be close)
        corpus_texts = [text for text, _ in small_being.sorter.entries]
        last_rank = sorted(range(len(amps)), key=lambda i: -amps[i]).index(
            corpus_texts.index(last_text))
        assert last_rank < len(corpus_texts) // 2, \
            "Annealed walk should converge toward high-amplitude entries"

