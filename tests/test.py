"""
Test Suite for PLEDGE-KARMA

Tests all core components:
  - MRL encoder and dimensional divergence
  - Knowledge state estimation (BKT + forgetting)
  - Prerequisite graph construction
  - PLEDGE retrieval
  - Full pipeline

Run with: pytest tests/test.py -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mrl_encoder import MRLEncoder, MRLEmbedding
from knowledge_graph.graph_builder import KnowledgeGraphBuilder, ConceptNode, CorpusChunk
from karma.estimator import (
    KARMAEstimator, EbbinghausForgettingCurve,
    BayesianKnowledgeTracker, Interaction
)
from pledge.retriever import (
    PLEDGERetriever, CognitiveLoadEstimator, DepthModulator,
    SubmodularRetriever, RetrievalResult
)
from pipeline.pledge_karma_pipeline import PLEDGEKARMAPipeline, PipelineResponse


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def encoder_config():
    return {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "matryoshka_dims": [64, 128, 256, 512, 768],
        "full_dim": 768,
        "batch_size": 4,
        "device": "cpu"
    }


@pytest.fixture
def encoder(encoder_config):
    return MRLEncoder(encoder_config)


@pytest.fixture
def karma_config():
    return {
        "forgetting": {
            "base_stability": 1.0,
            "stability_increase_rate": 0.2,
            "min_retention": 0.1,
            "retrievability_threshold": 0.5
        },
        "bkt": {
            "p_init": 0.1,
            "p_transit": 0.15,
            "p_slip": 0.1,
            "p_guess": 0.2,
            "mastery_threshold": 0.95
        },
        "metacognitive": {
            "fluency_illusion_decay": 0.85,
            "mrl_divergence_threshold": 0.15,
            "gap_smoothing_window": 5
        }
    }


@pytest.fixture
def karma(karma_config):
    return KARMAEstimator(karma_config)


@pytest.fixture
def sample_concepts():
    return [
        ConceptNode(
            concept_id="concept_velocity",
            name="Velocity",
            description="Rate of change of position",
            source_chunk_ids=[],
            depth_level=0,
            chapter_order=10,
            subject="physics"
        ),
        ConceptNode(
            concept_id="concept_acceleration",
            name="Acceleration",
            description="Rate of change of velocity",
            source_chunk_ids=[],
            depth_level=0,
            chapter_order=20,
            subject="physics"
        ),
        ConceptNode(
            concept_id="concept_force",
            name="Force",
            description="F = ma, push or pull on object",
            source_chunk_ids=[],
            depth_level=1,
            chapter_order=30,
            subject="physics"
        ),
        ConceptNode(
            concept_id="concept_momentum",
            name="Momentum",
            description="p = mv, product of mass and velocity",
            source_chunk_ids=[],
            depth_level=1,
            chapter_order=40,
            subject="physics"
        )
    ]


@pytest.fixture
def sample_chunks(sample_concepts):
    return [
        CorpusChunk(
            chunk_id="chunk_0001",
            text="Velocity is the rate of change of position with respect to time.",
            concept_ids=["concept_velocity"],
            prerequisite_concept_ids=[],
            depth_level=0,
            chapter_order=10,
            subject="physics",
            source="mock"
        ),
        CorpusChunk(
            chunk_id="chunk_0002",
            text="Acceleration is defined as dv/dt, the rate at which velocity changes.",
            concept_ids=["concept_acceleration"],
            prerequisite_concept_ids=["concept_velocity"],
            depth_level=0,
            chapter_order=20,
            subject="physics",
            source="mock"
        ),
        CorpusChunk(
            chunk_id="chunk_0003",
            text="Force equals mass times acceleration: F = ma. This is Newton's second law.",
            concept_ids=["concept_force"],
            prerequisite_concept_ids=["concept_acceleration"],
            depth_level=1,
            chapter_order=30,
            subject="physics",
            source="mock"
        ),
        CorpusChunk(
            chunk_id="chunk_0004",
            text="Momentum p = mv is conserved in closed systems. Advanced treatment using F=dp/dt.",
            concept_ids=["concept_momentum"],
            prerequisite_concept_ids=["concept_velocity", "concept_force"],
            depth_level=2,
            chapter_order=40,
            subject="physics",
            source="mock"
        ),
        CorpusChunk(
            chunk_id="chunk_0005",
            text="Basic introduction to velocity: speed with direction. Common units are m/s.",
            concept_ids=["concept_velocity"],
            prerequisite_concept_ids=[],
            depth_level=0,
            chapter_order=10,
            subject="physics",
            source="mock"
        ),
    ]


@pytest.fixture
def graph_config():
    return {
        "prerequisite_sim_threshold": 0.75,
        "cross_scale_agreement_threshold": 0.60,
        "min_edge_confidence": 0.65
    }


# ─────────────────────────────────────────────────────────────────────────────
# MRL Encoder Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMRLEncoder:

    def test_encode_single_text(self, encoder):
        emb = encoder.encode("What is velocity?", prompt_name="search_query")
        assert isinstance(emb, MRLEmbedding)
        assert emb.full_embedding.shape == (768,)

    def test_encode_batch(self, encoder):
        texts = ["velocity definition", "acceleration formula", "force law"]
        embs = encoder.encode(texts, prompt_name="search_document")
        assert len(embs) == 3
        for emb in embs:
            assert emb.full_embedding.shape == (768,)

    def test_at_dim_normalization(self, encoder):
        emb = encoder.encode("test text")
        for dim in [64, 128, 256, 512, 768]:
            vec = emb.at_dim(dim)
            assert vec.shape == (dim,)
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-5, f"Embedding at dim {dim} not normalized"

    def test_invalid_dim_raises(self, encoder):
        emb = encoder.encode("test")
        with pytest.raises(ValueError):
            emb.at_dim(100)  # Not in registered dims

    def test_similarity_range(self, encoder):
        emb_a = encoder.encode("velocity of a car")
        emb_b = encoder.encode("velocity definition in physics")
        for dim in [64, 768]:
            sim = encoder.compute_similarity(emb_a, emb_b, dim)
            assert -1.0 <= sim <= 1.0

    def test_dimensional_divergence(self, encoder):
        emb_a = encoder.encode_query("what is quantum entanglement?")
        emb_b = encoder.encode_document("velocity is the rate of change of position")
        div = encoder.compute_dimensional_divergence(emb_a, emb_b)
        assert isinstance(div, float)
        # Divergence can be any value in [-1, 1]
        assert -2.0 <= div <= 2.0

    def test_multiscale_agreement_range(self, encoder):
        emb_a = encoder.encode("physics force")
        emb_b = encoder.encode("physics acceleration")
        agreement = encoder.multiscale_agreement_score(emb_a, emb_b)
        assert 0.0 <= agreement <= 1.0

    def test_encode_query_vs_document(self, encoder):
        query_emb = encoder.encode_query("what is velocity")
        doc_emb = encoder.encode_document("velocity is the rate of change of position")
        # Both should produce valid MRLEmbeddings
        assert isinstance(query_emb, MRLEmbedding)
        assert isinstance(doc_emb, MRLEmbedding)


# ─────────────────────────────────────────────────────────────────────────────
# Forgetting Curve Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEbbinghausForgettingCurve:

    @pytest.fixture
    def forgetting(self):
        return EbbinghausForgettingCurve({
            "base_stability": 1.0,
            "stability_increase_rate": 0.2,
            "min_retention": 0.1,
            "retrievability_threshold": 0.5
        })

    def test_retention_at_zero_days(self, forgetting):
        r = forgetting.compute_retention(stability=2.0, days_since_review=0)
        assert r == 1.0

    def test_retention_decreases_with_time(self, forgetting):
        r1 = forgetting.compute_retention(20.0, 1)
        r2 = forgetting.compute_retention(20.0, 7)
        r3 = forgetting.compute_retention(20.0, 30)
        assert r1 > r2 > r3

    def test_retention_floor(self, forgetting):
        r = forgetting.compute_retention(0.1, 1000)
        assert r >= 0.1  # Min retention floor

    def test_stability_increases_on_success(self, forgetting):
        s0 = 2.0
        s_new = forgetting.update_stability(s0, days_since_review=3, success=True)
        assert s_new >= s0

    def test_stability_resets_on_failure(self, forgetting):
        s0 = 5.0  # High stability
        s_new = forgetting.update_stability(s0, days_since_review=3, success=False)
        assert s_new < s0  # Resets to base

    def test_needs_reactivation_after_long_gap(self, forgetting):
        assert forgetting.needs_reactivation(stability=1.0, days_since_review=30)

    def test_no_reactivation_needed_soon_after_review(self, forgetting):
        assert not forgetting.needs_reactivation(stability=5.0, days_since_review=1)


# ─────────────────────────────────────────────────────────────────────────────
# BKT Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBayesianKnowledgeTracker:

    @pytest.fixture
    def bkt(self):
        return BayesianKnowledgeTracker({
            "p_init": 0.1, "p_transit": 0.15,
            "p_slip": 0.1, "p_guess": 0.2,
            "mastery_threshold": 0.95
        })

    def test_correct_response_increases_mastery(self, bkt):
        p0 = 0.3
        p1 = bkt.update(p0, correct=True, response_quality=0.9)
        assert p1 > p0

    def test_incorrect_response_decreases_mastery(self, bkt):
        p0 = 0.8
        p1 = bkt.update(p0, correct=False, response_quality=0.1)
        assert p1 < p0

    def test_repeated_correct_converges_to_mastery(self, bkt):
        p = 0.1
        for _ in range(20):
            p = bkt.update(p, correct=True, response_quality=1.0)
        assert p > 0.8

    def test_mastery_probability_bounded(self, bkt):
        p = 0.5
        for _ in range(5):
            p = bkt.update(p, correct=True)
            assert 0.0 <= p <= 1.0

    def test_passive_update_without_assessment(self, bkt):
        p0 = 0.4
        p1 = bkt.update(p0, correct=None, response_quality=0.8)
        # Should be a mild update, still in valid range
        assert 0.0 <= p1 <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# KARMA Estimator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKARMAEstimator:

    def test_initial_state_is_prior(self, karma):
        p_obj, p_sub, gap = karma.get_knowledge_state("new_concept")
        assert p_obj == 0.1  # p_init
        assert p_sub == 0.1
        assert abs(gap) < 1e-5

    def test_knowledge_increases_with_correct_interactions(self, karma):
        concept = "concept_velocity"
        base_dt = datetime.now()
        for i in range(5):
            interaction = Interaction(
                interaction_id=f"test_{i}",
                timestamp=base_dt + timedelta(minutes=i * 10),
                query="explain velocity",
                concept_ids=[concept],
                correct=True,
                response_quality=0.9,
                mrl_divergence=0.05
            )
            karma.update(interaction)

        p_obj, _, _ = karma.get_knowledge_state(concept)
        assert p_obj > 0.5

    def test_forgetting_over_time(self, karma):
        concept = "concept_force"
        # Learn the concept
        interaction = Interaction(
            interaction_id="learn",
            timestamp=datetime(2024, 1, 1),
            query="explain force",
            concept_ids=[concept],
            correct=True,
            response_quality=1.0,
            mrl_divergence=0.0
        )
        karma.update(interaction)
        p_after_learning, _, _ = karma.get_knowledge_state(concept)

        # Advance time by 30 days
        karma.current_time = datetime(2024, 2, 1)
        p_after_forgetting, _, _ = karma.get_knowledge_state(concept)

        assert p_after_forgetting < p_after_learning

    def test_subjective_decays_slower_than_objective(self, karma):
        concept = "concept_entropy"
        # Learn concept
        interaction = Interaction(
            interaction_id="learn",
            timestamp=datetime(2024, 1, 1),
            query="entropy definition",
            concept_ids=[concept],
            correct=True,
            response_quality=0.95,
            mrl_divergence=0.0
        )
        karma.update(interaction)

        # Advance time significantly
        karma.current_time = datetime(2024, 2, 15)
        p_obj, p_sub, gap = karma.get_knowledge_state(concept)

        # Subjective should be >= objective (fluency illusion)
        assert p_sub >= p_obj or abs(p_sub - p_obj) < 0.01

    def test_metacognitive_gap_computation(self, karma):
        concept = "concept_quantum"
        p_obj, p_sub, gap = karma.get_knowledge_state(concept)
        assert abs(gap - (p_sub - p_obj)) < 1e-6

    def test_admissibility_confidence_no_prereqs(self, karma):
        conf = karma.compute_admissibility_confidence([])
        assert conf == 1.0

    def test_admissibility_confidence_with_unlearned_prereqs(self, karma):
        conf = karma.compute_admissibility_confidence(
            ["unknown_concept_1", "unknown_concept_2"],
            delta_penalty=0.3
        )
        # Unlearned concepts → low admissibility
        assert conf < 0.5

    def test_admissibility_confidence_with_learned_prereqs(self, karma):
        prereq = "concept_prereq_learned"
        # Simulate learning
        for i in range(10):
            interaction = Interaction(
                interaction_id=f"prereq_{i}",
                timestamp=datetime.now(),
                query="prereq",
                concept_ids=[prereq],
                correct=True,
                response_quality=1.0,
                mrl_divergence=0.0
            )
            karma.update(interaction)

        conf = karma.compute_admissibility_confidence([prereq])
        assert conf > 0.5

    def test_metacognitive_profile(self, karma):
        profile = karma.get_metacognitive_profile()
        assert "calibration" in profile
        assert "avg_gap" in profile


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeGraph:

    @pytest.fixture
    def graph(self, graph_config, encoder, sample_concepts, sample_chunks):
        g = KnowledgeGraphBuilder(graph_config, encoder)
        for concept in sample_concepts:
            rng = np.random.RandomState(hash(concept.concept_id) % 10000)
            concept.embedding = rng.randn(768).astype(np.float32)
            concept.embedding /= np.linalg.norm(concept.embedding)
            g.add_concept(concept)
        for chunk in sample_chunks:
            g.add_chunk(chunk)
        return g

    def test_concepts_added_to_graph(self, graph, sample_concepts):
        assert len(graph.concepts) == len(sample_concepts)

    def test_chunks_registered(self, graph, sample_chunks):
        assert len(graph.chunks) == len(sample_chunks)

    def test_prerequisite_edges_from_annotations(self, graph):
        edges = [
            ("concept_velocity", "concept_acceleration", 0.9),
            ("concept_acceleration", "concept_force", 0.85)
        ]
        n = graph.build_prerequisite_edges_from_annotations(edges)
        assert n == 2
        assert graph.graph.has_edge("concept_velocity", "concept_acceleration")

    def test_get_prerequisites(self, graph):
        graph.build_prerequisite_edges_from_annotations([
            ("concept_velocity", "concept_acceleration", 0.9),
            ("concept_acceleration", "concept_force", 0.85),
            ("concept_force", "concept_momentum", 0.8)
        ])
        prereqs = graph.get_prerequisites("concept_momentum", depth=2)
        assert "concept_force" in prereqs
        assert "concept_acceleration" in prereqs

    def test_zpd_identifies_frontier_concepts(self, graph):
        graph.build_prerequisite_edges_from_annotations([
            ("concept_velocity", "concept_acceleration", 0.9),
            ("concept_acceleration", "concept_force", 0.85)
        ])
        # Student knows velocity but not acceleration
        known = {"concept_velocity": 0.9}
        zpd = graph.get_zone_of_proximal_development(known)
        assert len(zpd) > 0
        # Acceleration should be in ZPD (prereq velocity is mastered)
        zpd_ids = [cid for cid, _ in zpd]
        assert "concept_acceleration" in zpd_ids

    def test_graph_summary(self, graph):
        summary = graph.summary()
        assert "n_concepts" in summary
        assert "n_chunks" in summary
        assert "n_edges" in summary
        assert summary["n_concepts"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Cognitive Load and Depth Modulator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCognitiveLoad:

    def test_known_concept_has_lower_load(self, karma, sample_chunks, graph_config, encoder, sample_concepts):
        g = KnowledgeGraphBuilder(graph_config, encoder)
        for c in sample_concepts:
            g.add_concept(c)
        for ch in sample_chunks:
            g.add_chunk(ch)

        cl_estimator = CognitiveLoadEstimator({
            "novel_concept_cost": 1.0,
            "dependency_depth_cost": 0.5,
            "working_memory_budget": 7.0
        })

        # Before learning
        load_before = cl_estimator.compute(sample_chunks[0], karma, g)

        # After learning velocity concept
        for i in range(5):
            interaction = Interaction(
                interaction_id=f"cl_{i}",
                timestamp=datetime.now(),
                query="velocity",
                concept_ids=["concept_velocity"],
                correct=True,
                response_quality=1.0,
                mrl_divergence=0.0
            )
            karma.update(interaction)

        load_after = cl_estimator.compute(sample_chunks[0], karma, g)
        assert load_after <= load_before


class TestDepthModulator:

    def test_low_mastery_gives_depth_zero(self, karma, graph_config, encoder, sample_concepts):
        g = KnowledgeGraphBuilder(graph_config, encoder)
        for c in sample_concepts:
            g.add_concept(c)

        dm = DepthModulator({"num_levels": 3, "depth_mismatch_penalty": 0.4})
        # Unlearned concepts → should target depth 0
        related = ["concept_velocity", "concept_acceleration"]
        target = dm.compute_target_depth(related, karma, g)
        assert target < 1.0

    def test_depth_match_score_perfect_match(self):
        dm = DepthModulator({"num_levels": 3, "depth_mismatch_penalty": 0.4})
        chunk = CorpusChunk(
            chunk_id="test", text="test", concept_ids=[], prerequisite_concept_ids=[],
            depth_level=1, chapter_order=0, subject="physics", source="test"
        )
        score = dm.compute_depth_match_score(chunk, target_depth=1.0)
        assert score > 0.9

    def test_depth_match_score_mismatch(self):
        dm = DepthModulator({"num_levels": 3, "depth_mismatch_penalty": 0.4})
        chunk = CorpusChunk(
            chunk_id="test", text="test", concept_ids=[], prerequisite_concept_ids=[],
            depth_level=2, chapter_order=0, subject="physics", source="test"
        )
        score = dm.compute_depth_match_score(chunk, target_depth=0.0)
        assert score < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Submodular Retriever Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSubmodularRetriever:

    def test_greedy_selects_k_items(self, encoder):
        sub = SubmodularRetriever({"diversity_weight": 0.2})
        rng = np.random.RandomState(42)

        # Create mock embeddings
        candidates = []
        for i in range(20):
            emb_array = rng.randn(768).astype(np.float32)
            emb_array /= np.linalg.norm(emb_array)
            emb = MRLEmbedding(full_embedding=emb_array, dims=[64,128,256,512,768])
            candidates.append((f"candidate_{i}", emb, 1.0))

        # Mock query embedding
        query_emb = encoder.encode("what is velocity?", prompt_name="search_query")
        selected = sub.greedy_select(candidates, query_emb, encoder, k=5)
        assert len(selected) == 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in selected)
        assert all(isinstance(item[0], str) for item in selected)
        assert all(isinstance(item[1], (int, float)) for item in selected)

    def test_marginal_gain_computation(self, encoder):
        sub = SubmodularRetriever({"diversity_weight": 0.2})
        rng = np.random.RandomState(43)
        query_emb = encoder.encode("what is velocity?", prompt_name="search_query")
        cand_emb = encoder.encode("velocity is the rate of change of position", prompt_name="search_document")
        selected_embs = [
            encoder.encode(f"document about physics {i}", prompt_name="search_document")
            for i in range(2)
        ]
        mg_empty = sub.compute_marginal_gain(cand_emb, query_emb, [], encoder, dim=768)
        assert isinstance(mg_empty, (int, float))
        assert -2.0 <= mg_empty <= 2.0
        mg_with_selected = sub.compute_marginal_gain(cand_emb, query_emb, selected_embs, encoder, dim=768)
        assert isinstance(mg_with_selected, (int, float))
        assert -2.0 <= mg_with_selected <= 2.0


# ─────────────────────────────────────────────────────────────────────────────
# PLEDGE Retriever Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPLEDGERetriever:

    @pytest.fixture
    def retriever_config(self):
        return {
            "retrieval": {
                "candidate_pool_size": 50,
                "final_k": 5,
                "diversity_weight": 0.2
            },
            "admissibility": {"confidence_threshold": 0.80},
            "depth": {"num_levels": 3, "depth_mismatch_penalty": 0.4},
            "cognitive_load": {
                "novel_concept_cost": 1.0,
                "dependency_depth_cost": 0.5,
                "working_memory_budget": 7.0
            }
        }

    def test_build_index_and_retrieve(
        self, encoder, karma, graph_config, sample_concepts, sample_chunks, retriever_config
    ):
        pytest.importorskip("faiss")
        graph = KnowledgeGraphBuilder(graph_config, encoder)
        for c in sample_concepts:
            rng = np.random.RandomState(hash(c.concept_id) % 10000)
            c.embedding = rng.randn(768).astype(np.float32)
            c.embedding /= np.linalg.norm(c.embedding)
            graph.add_concept(c)
        for ch in sample_chunks:
            graph.add_chunk(ch)

        retriever = PLEDGERetriever(retriever_config, encoder, graph, karma)
        chunks_list = list(graph.chunks.values())
        retriever.build_index(chunks_list, show_progress=False)

        result = retriever.retrieve("what is velocity?", student_concept_history=[])
        assert isinstance(result, RetrievalResult)
        assert result.query == "what is velocity?"
        assert hasattr(result, "retrieved_chunks")
        assert hasattr(result, "student_depth_level")
        assert hasattr(result, "admissibility_violations")
        assert hasattr(result, "reactivation_needed")
        assert hasattr(result, "metacognitive_profile")


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPLEDGEKARMAPipeline:

    def test_pipeline_answer_returns_response(
        self, encoder, karma, graph_config, sample_concepts, sample_chunks
    ):
        pytest.importorskip("faiss")
        graph = KnowledgeGraphBuilder(graph_config, encoder)
        for c in sample_concepts:
            rng = np.random.RandomState(hash(c.concept_id) % 10000)
            c.embedding = rng.randn(768).astype(np.float32)
            c.embedding /= np.linalg.norm(c.embedding)
            graph.add_concept(c)
        for ch in sample_chunks:
            graph.add_chunk(ch)

        retriever_config = {
            "retrieval": {"candidate_pool_size": 50, "final_k": 5, "diversity_weight": 0.2},
            "admissibility": {"confidence_threshold": 0.80},
            "depth": {"num_levels": 3, "depth_mismatch_penalty": 0.4},
            "cognitive_load": {
                "novel_concept_cost": 1.0,
                "dependency_depth_cost": 0.5,
                "working_memory_budget": 7.0
            }
        }
        retriever = PLEDGERetriever(retriever_config, encoder, graph, karma)
        retriever.build_index(list(graph.chunks.values()), show_progress=False)

        pipeline_config = {"karma": karma.config}
        pipeline = PLEDGEKARMAPipeline(
            pipeline_config, encoder, graph, karma, retriever, llm_client=None
        )
        response = pipeline.answer("what is velocity?")
        assert isinstance(response, PipelineResponse)
        assert response.query == "what is velocity?"
        assert hasattr(response, "answer")
        assert hasattr(response, "retrieval_result")
        assert hasattr(response, "depth_level_used")
        assert response.interaction_id.startswith("interaction_")
