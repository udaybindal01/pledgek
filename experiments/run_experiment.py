"""
Main Experiment Runner for PLEDGE-KARMA
See inline docs for usage.
"""
import argparse
import json
import logging
import os
import sys
import yaml
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mrl_encoder import MRLEncoder
from knowledge_graph.graph_builder import KnowledgeGraphBuilder, ConceptNode, CorpusChunk
from karma.estimator import KARMAEstimator, Interaction
from pledge.retriever import PLEDGERetriever
from pipeline.pledge_karma_pipeline import PLEDGEKARMAPipeline
from evaluation.evaluator import PLEDGEKARMAEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pledge_karma.log")]
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    import random; random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass


def build_mock_corpus(n_concepts=50, n_chunks=200):
    rng = np.random.RandomState(42)
    concept_data = [
        ("Velocity", "Rate of change of position with respect to time", 0, 10),
        ("Acceleration", "Rate of change of velocity with respect to time", 0, 20),
        ("Force", "A push or pull on an object; F = ma", 0, 30),
        ("Mass", "Measure of an object's resistance to acceleration", 0, 40),
        ("Momentum", "Product of mass and velocity, p = mv", 1, 50),
        ("Kinetic Energy", "Energy of motion, KE = 0.5mv^2", 1, 60),
        ("Potential Energy", "Stored energy due to position or configuration", 1, 70),
        ("Work", "Force applied over a distance, W = F·d·cos(θ)", 1, 80),
        ("Power", "Rate of doing work, P = W/t", 1, 90),
        ("Torque", "Rotational equivalent of force, τ = r × F", 2, 100),
        ("Angular Momentum", "Rotational analog of linear momentum", 2, 110),
        ("Gravity", "Attractive force between masses, F = Gm₁m₂/r²", 1, 120),
        ("Friction", "Resistive force opposing relative motion", 1, 130),
        ("Pressure", "Force per unit area, P = F/A", 1, 140),
        ("Temperature", "Average kinetic energy of particles in a system", 1, 150),
        ("Heat", "Transfer of thermal energy between objects", 1, 160),
        ("Entropy", "Measure of disorder in a thermodynamic system", 2, 170),
        ("Electric Field", "Region around a charge where forces act", 2, 180),
        ("Wavelength", "Distance between successive wave peaks", 1, 190),
        ("Quantum State", "Complete description of a quantum system", 2, 200),
    ]
    
    concepts = []
    prereq_chains = {
        "Momentum": ["Velocity", "Mass"],
        "Kinetic Energy": ["Velocity", "Mass"],
        "Work": ["Force"],
        "Power": ["Work"],
        "Torque": ["Force"],
        "Angular Momentum": ["Momentum", "Torque"],
        "Entropy": ["Heat", "Temperature"],
    }
    
    for name, desc, depth, order in concept_data[:n_concepts]:
        concept = ConceptNode(
            concept_id=f"concept_{name.lower().replace(' ', '_')}",
            name=name, description=desc, source_chunk_ids=[],
            depth_level=depth, chapter_order=order, subject="physics"
        )
        concept.embedding = rng.randn(768).astype(np.float32)
        concept.embedding /= np.linalg.norm(concept.embedding) + 1e-9
        concepts.append(concept)

    name_to_id = {c.name: c.concept_id for c in concepts}
    
    depth_templates = [
        "This is a basic introduction to {name}. {desc} Think of it as {analogy}.",
        "At an intermediate level, {name} is formally defined as: {desc} This connects to related topics.",
        "Advanced treatment: {name} — {desc} This has deep theoretical implications in modern physics.",
    ]
    analogies = ["everyday motion you can see", "energy you can feel", "forces you experience daily"]
    
    chunks = []
    chunk_counter = 0
    for concept in concepts:
        prereqs = [name_to_id[p] for p in prereq_chains.get(concept.name, []) if p in name_to_id]
        for depth in range(3):
            if chunk_counter >= n_chunks:
                break
            text = depth_templates[depth].format(
                name=concept.name, desc=concept.description,
                analogy=analogies[chunk_counter % len(analogies)]
            )
            chunk = CorpusChunk(
                chunk_id=f"chunk_{chunk_counter:04d}",
                text=text, concept_ids=[concept.concept_id],
                prerequisite_concept_ids=prereqs,
                depth_level=depth, chapter_order=concept.chapter_order,
                subject="physics", source="mock"
            )
            concept.source_chunk_ids.append(chunk.chunk_id)
            chunks.append(chunk)
            chunk_counter += 1
        if chunk_counter >= n_chunks:
            break

    logger.info(f"Mock corpus: {len(concepts)} concepts, {len(chunks)} chunks")
    return concepts, chunks


def build_graph(concepts, chunks, config, encoder,
               external_prereq_edges=None):
    """
    Build the prerequisite knowledge graph.

    Three edge sources (applied in order, later sources supplement earlier):
      1. Chunk-embedded prereqs  — from corpus structure (always fast)
      2. External annotations    — LectureBank / MOOCCube (if provided)
      3. Chapter-ordering edges  — automated from chapter position + MRL
         (with fixed gap threshold: 3 chapters, not 5 raw units)

    external_prereq_edges: list of (src_id, dst_id, confidence) or None.
    """
    graph = KnowledgeGraphBuilder(config.get("knowledge_graph", {}), encoder)
    for c in concepts: graph.add_concept(c)
    for ch in chunks:  graph.add_chunk(ch)

    # ── Step 1: annotation edges from corpus chunk structure ────────────────
    chunk_annotation_edges = []
    for chunk in chunks:
        for prereq_id in chunk.prerequisite_concept_ids:
            for concept_id in chunk.concept_ids:
                if prereq_id != concept_id:
                    chunk_annotation_edges.append((prereq_id, concept_id, 0.85))
    if chunk_annotation_edges:
        n = graph.build_prerequisite_edges_from_annotations(chunk_annotation_edges)
        logger.info(f"  {n} edges from chunk annotations")

    # ── Step 2: external human annotations (LectureBank / MOOCCube) ────────
    if external_prereq_edges:
        n = graph.build_prerequisite_edges_from_annotations(external_prereq_edges)
        logger.info(f"  {n} edges from external annotations")

    # ── Step 3: chapter-ordering edges ─────────────────────────────────────
    # The gap threshold was previously hardcoded to 5 raw units, which killed
    # all cross-chapter edges (OpenStax encodes chapter_order = ch*100+page).
    # Fixed in graph_builder.py: now uses chapter_gap = order_gap // 100 <= 3.
    # This is the primary edge source for OpenStax data with mock encoder.
    n_with_emb = sum(1 for c in concepts if c.embedding is not None)
    if n_with_emb > 0:
        n = graph.build_prerequisite_edges_from_ordering()
        logger.info(f"  {n} edges from chapter ordering")
    else:
        logger.warning("  No embeddings on concepts — skipping ordering edges.")
        logger.warning("  Install sentence_transformers for real edges on large corpora.")

    # ── Step 4: propagate concept prereqs onto chunks ───────────────────────
    # CRITICAL: without this, chunk.prerequisite_concept_ids stays empty
    # and PLEDGE's admissibility filter is permanently disabled.
    n_backfilled = graph.backfill_chunk_prerequisites(min_confidence=0.50)
    logger.info(f"  Backfilled prereqs for {n_backfilled} chunks")

    summary = graph.summary()
    logger.info(f"Graph: {summary}")
    if summary.get("n_edges", 0) == 0:
        logger.warning(
            "Graph has 0 edges! Admissibility will be trivially 1.0. "
            "Install sentence_transformers to get real embeddings and edges."
        )
    return graph


def get_default_config():
    return {
        "encoder": {"model_name": "nomic-ai/nomic-embed-text-v1.5",
                    "matryoshka_dims": [64, 128, 256, 512, 768],
                    "full_dim": 768, "batch_size": 32, "device": "cpu"},
        "knowledge_graph": {"prerequisite_sim_threshold": 0.75,
                            "cross_scale_agreement_threshold": 0.60,
                            "min_edge_confidence": 0.35},
        "karma": {
            "forgetting": {"base_stability": 1.0, "stability_increase_rate": 0.2,
                           "min_retention": 0.1, "retrievability_threshold": 0.5},
            "bkt": {"p_init": 0.1, "p_transit": 0.15, "p_slip": 0.1,
                    "p_guess": 0.2, "mastery_threshold": 0.95},
            "metacognitive": {"fluency_illusion_decay": 0.85,
                              "mrl_divergence_threshold": 0.15,
                              "gap_smoothing_window": 5}
        },
        "pledge": {
            "admissibility": {"hard_constraint": False,
                              "confidence_threshold": 0.80,
                              "uncertainty_penalty": 0.3},
            "depth": {"num_levels": 3, "depth_mismatch_penalty": 0.4},
            "retrieval": {"candidate_pool_size": 50, "final_k": 5,
                          "lambda_cognitive_load": 0.3,
                          "lambda_reactivation": 0.4,
                          "diversity_weight": 0.2,
                          "submodular_greedy_steps": 5},
            "cognitive_load": {"novel_concept_cost": 1.0,
                               "dependency_depth_cost": 0.5,
                               "working_memory_budget": 7.0}
        },
        "logging": {"output_dir": "outputs"},
        "project": {"seed": 42}
    }


def run_demo(config):
    logger.info("=== PLEDGE-KARMA Demo ===")
    encoder = MRLEncoder(config["encoder"])
    concepts, chunks = build_mock_corpus(n_concepts=20, n_chunks=60)
    graph = build_graph(concepts, chunks, config, encoder)
    karma = KARMAEstimator(config["karma"])
    retriever = PLEDGERetriever(config["pledge"], encoder, graph, karma)
    retriever.build_index(chunks)
    pipeline = PLEDGEKARMAPipeline(config, encoder, graph, karma, retriever, llm_client=None)

    queries = [
        "What is velocity?",
        "How does force relate to acceleration?",
        "Can you explain entropy?"
    ]

    print("\n--- WEEK 2 STUDENT (Beginner) ---")
    for query in queries[:2]:
        resp = pipeline.answer(query)
        print(f"\nQ: {query}")
        print(f"  Mode: {resp.response_mode} | Depth: {resp.depth_level_used}")
        print(f"  Chunks: {len(resp.retrieval_result.retrieved_chunks)} | "
              f"Violations: {resp.retrieval_result.admissibility_violations}")

    # Simulate 8 weeks of learning
    base_dt = datetime.now()
    for week in range(8):
        for concept in concepts[:12]:
            interaction = Interaction(
                interaction_id=f"w{week}_{concept.concept_id}",
                timestamp=base_dt + timedelta(weeks=week),
                query=f"explain {concept.name}",
                concept_ids=[concept.concept_id],
                correct=(week > 3),
                response_quality=min(0.9, 0.3 + 0.08 * week),
                mrl_divergence=max(0.05, 0.2 - 0.02 * week)
            )
            karma.update(interaction)
    karma.current_time = base_dt + timedelta(weeks=8)

    print("\n--- WEEK 8 STUDENT (Same queries, should see different depth/mode) ---")
    for query in queries[:2]:
        resp = pipeline.answer(query)
        print(f"\nQ: {query}")
        print(f"  Mode: {resp.response_mode} | Depth: {resp.depth_level_used}")
        print(f"  Metacognitive: {resp.metadata.get('metacognitive_profile', {}).get('calibration', 'N/A')}")

    print("\n=== Demo complete. PLEDGE-KARMA adapts depth and mode to student state. ===")


def run_evaluation(config, mode="quick"):
    logger.info(f"=== Evaluation Mode: {mode} ===")
    seed_everything(config["project"]["seed"])
    
    n_students = 15 if mode == "quick" else 100
    n_weeks = 4 if mode == "quick" else 10
    n_concepts = 20 if mode == "quick" else 50
    n_chunks = 60 if mode == "quick" else 200

    encoder = MRLEncoder(config["encoder"])
    concepts, chunks = build_mock_corpus(n_concepts=n_concepts, n_chunks=n_chunks)
    graph = build_graph(concepts, chunks, config, encoder)
    
    karma = KARMAEstimator(config["karma"])
    retriever = PLEDGERetriever(config["pledge"], encoder, graph, karma)
    retriever.build_index(chunks)

    chunk_map = {c.chunk_id: c for c in chunks}
    available_ids = list(retriever._chunk_embeddings.keys())
    emb_matrix = np.stack([
        retriever._chunk_embeddings[cid].at_dim(768) for cid in available_ids
    ]).astype(np.float32)

    def standard_rag(query, karma, target_concepts, k=5):
        q = encoder.encode_query(query).at_dim(768).reshape(1, -1)
        sims = (q @ emb_matrix.T).flatten()
        top = np.argsort(sims)[::-1][:k]
        ids = [available_ids[i] for i in top]
        return ids, [chunk_map[i] for i in ids if i in chunk_map]

    # Pre-build 64D embedding matrix for pledge_naive_kt (one-time cost)
    emb64_matrix = np.stack([
        retriever._chunk_embeddings[cid].at_dim(64) for cid in available_ids
    ]).astype(np.float32)

    def pledge_naive_kt(query, karma, target_concepts, k=5):
        """
        PLEDGE with naive K_t: 768D retrieval + prerequisite admissibility filter.
        Uses same mastery threshold (0.60) as the evaluator so admissibility_rate
        is computed on a consistent scale.
        Does NOT use metacognitive gap correction (the key KARMA contribution).
        """
        q768 = encoder.encode_query(query).at_dim(768).reshape(1, -1).astype(np.float32)
        sims = (q768 @ emb_matrix.T).flatten()
        top_k3 = np.argsort(sims)[::-1][:k * 6]  # wider pool to survive filtering
        filtered = []
        for idx in top_k3:
            cid   = available_ids[idx]
            chunk = chunk_map.get(cid)
            if chunk:
                # Match evaluator threshold of 0.60
                ok = all(
                    karma.get_knowledge_state(p)[0] >= 0.60
                    for p in chunk.prerequisite_concept_ids
                )
                if ok:
                    filtered.append(cid)
        # Soft fallback: no-prereq chunks are always admissible (introductory content)
        if not filtered:
            for idx in top_k3:
                cid   = available_ids[idx]
                chunk = chunk_map.get(cid)
                if chunk and not chunk.prerequisite_concept_ids:
                    filtered.append(cid)
        filtered = (filtered or [available_ids[i] for i in top_k3])[:k]
        return filtered, [chunk_map[c] for c in filtered if c in chunk_map]

    def pledge_karma_full(query, karma, target_concepts, k=5):
        """
        Full PLEDGE-KARMA: FAISS retrieval + depth-aware scoring + KARMA
        metacognitive correction for admissibility filtering.
        """
        # Reuse the shared FAISS index; only swap the KARMA estimator
        retriever._karma = karma
        result = retriever.retrieve(query)
        ids = [rc.chunk.chunk_id for rc in result.retrieved_chunks]
        return ids, [rc.chunk for rc in result.retrieved_chunks]

    methods = {
        "standard_rag": standard_rag,
        "pledge_naive_kt": pledge_naive_kt,
        "pledge_karma_full": pledge_karma_full,
    }

    os.makedirs("outputs", exist_ok=True)
    evaluator = PLEDGEKARMAEvaluator(config, graph)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = evaluator.compare_methods(
        methods=methods,
        n_students=n_students,
        n_weeks=n_weeks,
        karma_config=config["karma"],
        output_path=f"outputs/eval_{mode}_{ts}.json"
    )
    return results


def run_ablation(config):
    logger.info("=== Critical Ablation: Does KARMA help PLEDGE? ===")
    seed_everything(42)
    encoder = MRLEncoder(config["encoder"])
    concepts, chunks = build_mock_corpus(n_concepts=25, n_chunks=80)
    graph = build_graph(concepts, chunks, config, encoder)
    karma = KARMAEstimator(config["karma"])
    retriever = PLEDGERetriever(config["pledge"], encoder, graph, karma)
    retriever.build_index(chunks)

    chunk_map = {c.chunk_id: c for c in chunks}
    available_ids = list(retriever._chunk_embeddings.keys())
    emb768 = np.stack([retriever._chunk_embeddings[cid].at_dim(768) for cid in available_ids])
    emb64 = np.stack([retriever._chunk_embeddings[cid].at_dim(64) for cid in available_ids])

    def pledge_naive_kt(query, karma, target_concepts, k=5):
        q = encoder.encode_query(query).at_dim(768).reshape(1, -1)
        sims = (q @ emb768.T).flatten()
        top = [available_ids[i] for i in np.argsort(sims)[::-1][:k]]
        return top, [chunk_map[i] for i in top if i in chunk_map]

    def pledge_karma_full(query, karma, target_concepts, k=5):
        fresh = PLEDGERetriever(config["pledge"], encoder, graph, karma)
        fresh._index_built = True
        fresh.faiss_index = retriever.faiss_index
        fresh._chunk_embeddings = retriever._chunk_embeddings
        result = fresh.retrieve(query)
        ids = [rc.chunk.chunk_id for rc in result.retrieved_chunks]
        return ids, [rc.chunk for rc in result.retrieved_chunks]

    evaluator = PLEDGEKARMAEvaluator(config, graph)
    results = evaluator.compare_methods(
        {"pledge_naive_kt": pledge_naive_kt, "pledge_karma_full": pledge_karma_full},
        n_students=30, n_weeks=6, karma_config=config["karma"]
    )

    naive = results["pledge_naive_kt"]
    full = results["pledge_karma_full"]
    print("\n=== ABLATION VERDICT ===")
    metrics = ["admissibility_rate", "depth_accuracy", "ndcg_at_10",
               "simulated_learning_gain"]
    wins = 0
    for m in metrics:
        delta = getattr(full, m) - getattr(naive, m)
        wins += (delta > 0)
        print(f"  {m}: {'↑' if delta > 0 else '↓'} {abs(delta):.4f}")
    print(f"\n{'✓ Combine PLEDGE+KARMA (justified)' if wins >= 3 else '⚠ Consider separate papers'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--mode", choices=["demo", "quick", "full", "ablation"], default="demo")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()

    os.makedirs("outputs", exist_ok=True)

    if args.mode == "demo":
        run_demo(config)
    elif args.mode in ("full", "quick"):
        run_evaluation(config, mode=args.mode)
    elif args.mode == "ablation":
        run_ablation(config)


if __name__ == "__main__":
    main()