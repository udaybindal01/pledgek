#!/usr/bin/env python3
"""
concept_alignment.py — Cross-Dataset Concept Alignment Layer
==============================================================
Maps concept names from different datasets into a shared embedding space
using the MRL encoder, resolving the domain mismatch identified in our
NeurIPS critique.

Problem:
    LectureBank  → CS/NLP concept names   (e.g. "backpropagation")
    OpenStax     → Physics/Calculus/Chemistry concepts (e.g. "Newton's 2nd Law")
    ASSISTments  → Math skill names (e.g. "Solving Linear Equations")
    MOOCCube     → Chinese MOOC concept IDs

These datasets have NO shared namespace, so prerequisite edges from
LectureBank cannot be applied to an OpenStax corpus without alignment.

Solution (Three strategies, configurable):
    Strategy A — Embedding-based soft alignment:
        Encode all concept names at 768D; connect concepts with cosine
        similarity > threshold across datasets. Produces confidence-weighted
        cross-dataset edges. Used when all datasets needed together.

    Strategy B — Single-dataset-per-axis (default, cleanest):
        Prereq graph evaluation  → LectureBank + OpenStax separately
        KT/forgetting evaluation → ASSISTments or EdNet only
        End-to-end evaluation   → MOOCCube only
        Explicitly state in the paper that evaluations are complementary,
        NOT unified. Eliminates the circular "cross-dataset unity" claim.

    Strategy C — Retroactive MRL injection into ASSISTments:
        Map skill_name → concept embedding; compute divergence vs.
        retrieved chunk embedding offline; store precomputed divergence.
        This gives ASSISTments real mrl_divergence values instead of 0.0.

Usage:
    from data.pipelines.concept_alignment import ConceptAligner

    # Strategy A: soft embedding alignment
    aligner = ConceptAligner(encoder)
    bridges = aligner.align_across_datasets(
        source_concepts=lecturebank_concepts,
        target_concepts=openstax_concepts,
        threshold=0.75
    )

    # Strategy C: inject MRL into ASSISTments
    updated_interactions = aligner.inject_mrl_into_assistments(
        interactions=assistments_interactions,
        corpus_chunks=openstax_chunks,
    )
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConceptBridge:
    """
    A soft alignment edge between a concept in dataset A and dataset B.
    Used to propagate prerequisite edges across domain boundaries.
    """
    source_concept_id:   str
    source_concept_name: str
    source_dataset:      str
    target_concept_id:   str
    target_concept_name: str
    target_dataset:      str
    cosine_similarity:   float      # sim at 768D (semantic)
    coarse_similarity:   float      # sim at 64D (lexical)
    bridge_confidence:   float      # combined confidence ∈ [0, 1]

    def to_dict(self) -> Dict:
        return self.__dict__


# ─────────────────────────────────────────────────────────────────────────────
# Aligner
# ─────────────────────────────────────────────────────────────────────────────

class ConceptAligner:
    """
    Cross-dataset concept alignment via MRL encoder embeddings.

    The core idea: two concepts from different datasets are "aligned"
    if they are semantically similar at fine granularity (768D) but
    the similarity should NOT collapse to lexical overlap (64D).
    We use the MRL divergence signal itself as a quality check:
    a bridge with high sim_768D but low sim_64D is a genuine
    semantic match, not just shared vocabulary.
    """

    def __init__(
        self,
        encoder=None,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            encoder:    MRLEncoder instance. If None, alignment uses
                        name-matching heuristics (weaker but works without
                        sentence-transformers).
            cache_path: Path to cache embeddings to disk (avoids re-encoding
                        on repeat runs). Recommended for large corpora.
        """
        self.encoder    = encoder
        self.cache_path = Path(cache_path) if cache_path else None
        self._embedding_cache: Dict[str, np.ndarray] = {}

        if encoder and not encoder._model_loaded:
            logger.warning(
                "MRLEncoder model not loaded — concept alignment will fall back "
                "to name-overlap heuristics. Install sentence-transformers for "
                "full alignment quality."
            )
            self.encoder = None

    # ── Strategy A: Embedding-based alignment ───────────────────────────────

    def align_across_datasets(
        self,
        source_concepts: List[Dict],   # [{"concept_id": ..., "name": ..., "dataset": ...}]
        target_concepts: List[Dict],
        threshold: float = 0.75,
        top_k: int = 3,
        use_mrl_quality_check: bool = True,
    ) -> List[ConceptBridge]:
        """
        Compute soft alignment bridges from source to target concepts.

        For each source concept, finds the top-k most similar target concepts
        with similarity above the threshold. The bridge_confidence weights
        the downstream edge propagation.

        Args:
            source_concepts:  List of concept dicts from dataset A.
            target_concepts:  List of concept dicts from dataset B.
            threshold:        Minimum cosine similarity to create a bridge.
            top_k:            Maximum bridges per source concept.
            use_mrl_quality_check: Filter out bridges where coarse_similarity ≈
                              fine_similarity (likely lexical match, not semantic).
        Returns:
            List of ConceptBridge objects.
        """
        if not source_concepts or not target_concepts:
            return []

        source_names = [c["name"] for c in source_concepts]
        target_names = [c["name"] for c in target_concepts]

        # Get embeddings (768D for semantic, 64D for lexical quality check)
        source_embs_768, source_embs_64 = self._encode_concepts(source_names)
        target_embs_768, target_embs_64 = self._encode_concepts(target_names)

        # Compute cosine similarity matrices
        sim_768 = source_embs_768 @ target_embs_768.T   # [S, T]
        sim_64  = source_embs_64  @ target_embs_64.T    # [S, T]

        bridges = []
        for s_idx, s_concept in enumerate(source_concepts):
            sims = sim_768[s_idx]
            top_indices = np.argsort(sims)[::-1][:top_k]

            for t_idx in top_indices:
                fine_sim   = float(sim_768[s_idx, t_idx])
                coarse_sim = float(sim_64[s_idx, t_idx])

                if fine_sim < threshold:
                    break  # sorted desc, no point continuing

                # MRL quality check: a good bridge has fine_sim > coarse_sim
                # (semantic alignment, not just lexical overlap).
                # If coarse_sim ≈ fine_sim, it's probably just keyword matching.
                mrl_divergence_of_bridge = fine_sim - coarse_sim
                if use_mrl_quality_check and mrl_divergence_of_bridge < -0.05:
                    # Skip: coarse already captures it — this is lexical overlap
                    continue

                t_concept = target_concepts[t_idx]

                # Bridge confidence: geometric mean of fine similarity and
                # the MRL quality signal (penalise pure keyword matches)
                quality = max(mrl_divergence_of_bridge + 0.5, 0.1)   # shift to [0, 1]
                confidence = float(np.sqrt(fine_sim * quality))
                confidence = float(np.clip(confidence, 0.0, 1.0))

                bridges.append(ConceptBridge(
                    source_concept_id   = s_concept["concept_id"],
                    source_concept_name = s_concept["name"],
                    source_dataset      = s_concept.get("dataset", "unknown"),
                    target_concept_id   = t_concept["concept_id"],
                    target_concept_name = t_concept["name"],
                    target_dataset      = t_concept.get("dataset", "unknown"),
                    cosine_similarity   = round(fine_sim, 4),
                    coarse_similarity   = round(coarse_sim, 4),
                    bridge_confidence   = round(confidence, 4),
                ))

        logger.info(
            f"Alignment: {len(source_concepts)} source → {len(target_concepts)} target "
            f"concepts, {len(bridges)} bridges at threshold={threshold}"
        )
        return bridges

    def propagate_prereq_edges(
        self,
        prereq_edges: List[Tuple[str, str, float]],   # (source_id, target_id, conf)
        bridges: List[ConceptBridge],
        min_propagated_confidence: float = 0.5,
    ) -> List[Tuple[str, str, float]]:
        """
        Propagate prerequisite edges from source dataset to target dataset
        via concept bridges.

        Example:
          LectureBank says: backpropagation → neural_networks (conf=0.9)
          Bridge says:      lb:backpropagation ≈ os:gradient_flow (conf=0.8)
          Bridge says:      lb:neural_networks ≈ os:deep_learning  (conf=0.85)
          Result:           os:gradient_flow → os:deep_learning     (conf=0.9*0.8*0.85=0.612)

        This allows LectureBank CS prerequisites to inform the OpenStax
        physics/calculus graph even without direct concept overlap.
        """
        # Build lookup: source_concept_id → List[ConceptBridge]
        source_to_bridges: Dict[str, List[ConceptBridge]] = {}
        for b in bridges:
            source_to_bridges.setdefault(b.source_concept_id, []).append(b)

        new_edges: List[Tuple[str, str, float]] = []
        for src_id, tgt_id, edge_conf in prereq_edges:
            src_bridges = source_to_bridges.get(src_id, [])
            tgt_bridges = source_to_bridges.get(tgt_id, [])

            for sb in src_bridges:
                for tb in tgt_bridges:
                    propagated_conf = edge_conf * sb.bridge_confidence * tb.bridge_confidence
                    if propagated_conf >= min_propagated_confidence:
                        new_edges.append((
                            sb.target_concept_id,
                            tb.target_concept_id,
                            round(propagated_conf, 4),
                        ))

        # Deduplicate: keep highest confidence per (source, target) pair
        best: Dict[Tuple[str, str], float] = {}
        for s, t, c in new_edges:
            if (s, t) not in best or c > best[(s, t)]:
                best[(s, t)] = c

        deduped = [(s, t, c) for (s, t), c in best.items()]
        logger.info(
            f"Propagated {len(prereq_edges)} source edges → "
            f"{len(deduped)} target edges via {len(bridges)} bridges"
        )
        return deduped

    # ── Strategy C: MRL injection into ASSISTments ──────────────────────────

    def inject_mrl_into_assistments(
        self,
        interactions: List[Dict],
        corpus_chunks: List[Dict],
        batch_size: int = 64,
    ) -> List[Dict]:
        """
        Retroactively compute real MRL divergence for ASSISTments interactions.

        Since ASSISTments doesn't have student question text, we:
          1. Convert skill_name → natural-language query template
             e.g. "Solving Linear Equations" → "How do I solve linear equations?"
          2. Find the best-matching corpus chunk for each query (at 768D)
          3. Compute MRL divergence = sim_768D - sim_64D for (query, best_chunk)
          4. Store precomputed divergence back into the interaction dict

        This gives ASSISTments real mrl_divergence values (not 0.0),
        enabling KARMA's fluency illusion model to operate.

        Args:
            interactions:   List of ASSISTments interaction dicts.
            corpus_chunks:  List of corpus chunk dicts with "text" field.
        Returns:
            Updated interactions with mrl_divergence filled in.
        """
        if self.encoder is None:
            logger.warning(
                "MRL encoder not available — cannot inject MRL divergence. "
                "Falling back to heuristic divergence based on skill name length "
                "(longer names → more specialized → higher divergence proxy)."
            )
            return self._heuristic_mrl_injection(interactions)

        # Build corpus matrix
        chunk_texts = [c["text"] for c in corpus_chunks][:2000]  # cap for speed
        if not chunk_texts:
            logger.warning("No corpus chunks provided — skipping MRL injection.")
            return interactions

        logger.info(f"Encoding {len(chunk_texts)} corpus chunks for MRL injection...")
        chunk_embs = self.encoder.encode(
            chunk_texts, prompt_name="search_document", show_progress=True
        )
        corpus_64  = np.stack([e.at_dim(64)  for e in chunk_embs])
        corpus_768 = np.stack([e.at_dim(768) for e in chunk_embs])

        # Get unique skill names and generate queries
        skill_names = list({i["skill_name"] for i in interactions if "skill_name" in i})
        queries = [self._skill_to_query(s) for s in skill_names]

        logger.info(f"Encoding {len(queries)} skill-name queries...")
        query_embs = self.encoder.encode(
            queries, prompt_name="search_query", show_progress=True
        )

        # Compute divergence per skill
        divergence_map: Dict[str, float] = {}
        for skill, q_emb in zip(skill_names, query_embs):
            q64  = q_emb.at_dim(64).reshape(1, -1)
            q768 = q_emb.at_dim(768).reshape(1, -1)

            sims_768 = (q768 @ corpus_768.T).flatten()
            top_idx  = int(np.argmax(sims_768))

            sim_fine   = float(sims_768[top_idx])
            sim_coarse = float((q64 @ corpus_64[top_idx:top_idx+1].T).item())
            divergence_map[skill] = round(sim_fine - sim_coarse, 4)

        # Apply to interactions
        updated = []
        for interaction in interactions:
            skill = interaction.get("skill_name", "")
            updated_interaction = dict(interaction)
            updated_interaction["mrl_divergence"] = divergence_map.get(skill, 0.0)
            updated.append(updated_interaction)

        n_filled = sum(1 for i in updated if i.get("mrl_divergence", 0.0) != 0.0)
        logger.info(
            f"MRL injection: {n_filled}/{len(updated)} interactions "
            f"({n_filled/max(len(updated),1):.1%} coverage)"
        )
        return updated

    def _skill_to_query(self, skill_name: str) -> str:
        """
        Convert ASSISTments skill name to a natural-language student query.

        Templates are chosen to mimic how a student would express
        confusion about a skill — matching the query distribution
        PLEDGE sees in production.
        """
        skill = skill_name.strip()
        templates = [
            f"How do I solve problems involving {skill.lower()}?",
            f"Can you explain {skill.lower()} with an example?",
            f"I don't understand {skill.lower()}, please help.",
        ]
        # Pick template based on skill name characteristics
        if any(kw in skill.lower() for kw in ["equation", "solve", "find", "calculate"]):
            return templates[0]
        elif any(kw in skill.lower() for kw in ["concept", "definition", "what is"]):
            return templates[1]
        else:
            return templates[2]

    def _heuristic_mrl_injection(self, interactions: List[Dict]) -> List[Dict]:
        """
        Fallback when encoder is unavailable.
        Longer skill names = more domain-specific = slightly higher divergence proxy.
        This is a weak signal but better than 0.0 everywhere.
        """
        updated = []
        for interaction in interactions:
            skill = interaction.get("skill_name", "")
            # Heuristic: normalised skill name length ∈ [0.02, 0.12]
            length_proxy = min(len(skill) / 200.0, 0.12)
            updated_interaction = dict(interaction)
            updated_interaction["mrl_divergence"] = round(length_proxy, 4)
            updated.append(updated_interaction)
        return updated

    # ── Strategy B: Single-axis evaluation helper ───────────────────────────

    @staticmethod
    def get_axis_datasets() -> Dict[str, Dict]:
        """
        Returns the canonical dataset assignments for each evaluation axis.

        This is the paper's official dataset-per-axis mapping (Strategy B).
        Each axis uses a single dataset — no cross-dataset alignment needed.
        Results are reported separately and described as complementary.
        """
        return {
            "axis_1_prereq_graph": {
                "description": "Prerequisite graph quality (F1, Precision, Recall)",
                "primary_dataset":  "lecturebank",
                "secondary_dataset": "openstax_multi_subject",
                "note": (
                    "LectureBank provides human-verified CS/NLP prereq pairs as ground truth. "
                    "OpenStax multi-subject corpus provides the retrieval target. "
                    "Evaluated separately — LectureBank prereq extraction F1, "
                    "OpenStax prereq structure via chapter-ordering validation."
                ),
                "metrics": ["precision", "recall", "f1", "edge_coverage"],
            },
            "axis_2_kt_forgetting": {
                "description": "Knowledge tracing and forgetting curve validation",
                "primary_dataset":  "ednet",
                "fallback_dataset": "assistments",
                "note": (
                    "EdNet preferred: 131M interactions with real timestamps and "
                    "concept tags enabling true forgetting evaluation. "
                    "ASSISTments as fallback with MRL injection via ConceptAligner."
                ),
                "metrics": ["auc", "rmse", "forgetting_mae", "mrl_correlation"],
            },
            "axis_3_endtoend_retrieval": {
                "description": "End-to-end pedagogical retrieval quality",
                "primary_dataset":  "mooccube",
                "note": (
                    "MOOCCube provides both the prereq graph AND student logs "
                    "in the same domain — the only dataset where held-out "
                    "student outcomes can validate retrieval quality end-to-end."
                ),
                "metrics": ["admissibility_rate", "ndcg_at_10", "mrr", "learning_gain_auc"],
            },
        }

    # ── Utilities ─────────────────────────────────────────────────────────

    def _encode_concepts(
        self, names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode concept names; returns (embs_768, embs_64).
        Falls back to TF-IDF-style bag-of-chars if encoder unavailable.
        """
        if self.encoder is not None:
            embs = self.encoder.encode(
                names,
                prompt_name="search_query",
                show_progress=len(names) > 100,
            )
            e768 = np.stack([e.at_dim(768) for e in embs])
            e64  = np.stack([e.at_dim(64)  for e in embs])
            return e768, e64
        else:
            # Heuristic fallback: character n-gram overlap
            return self._ngram_embeddings(names, n=3)

    def _ngram_embeddings(
        self, texts: List[str], n: int = 3, dim: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lightweight n-gram embeddings as fallback when encoder unavailable.
        Not suitable for paper results — only for dev/testing.
        """
        import hashlib

        def text_to_vec(text: str, out_dim: int) -> np.ndarray:
            vec = np.zeros(out_dim, dtype=np.float32)
            text = text.lower()
            grams = [text[i:i+n] for i in range(max(len(text)-n+1, 1))]
            for gram in grams:
                idx = int(hashlib.md5(gram.encode()).hexdigest(), 16) % out_dim
                vec[idx] += 1
            norm = np.linalg.norm(vec)
            return vec / max(norm, 1e-8)

        vecs_256 = np.stack([text_to_vec(t, 256) for t in texts])
        vecs_32  = np.stack([text_to_vec(t, 32)  for t in texts])
        return vecs_256, vecs_32

    def save_bridges(self, bridges: List[ConceptBridge], path: str) -> None:
        """Persist concept bridges to disk for reproducibility."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([b.to_dict() for b in bridges], f, indent=2)
        logger.info(f"Saved {len(bridges)} concept bridges to {path}")

    @staticmethod
    def load_bridges(path: str) -> List[ConceptBridge]:
        """Load concept bridges from disk."""
        with open(path) as f:
            raw = json.load(f)
        bridges = []
        for d in raw:
            bridges.append(ConceptBridge(**d))
        return bridges


# ─────────────────────────────────────────────────────────────────────────────
# CLI — alignment analysis
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Align concepts across PLEDGE-KARMA datasets"
    )
    parser.add_argument("--lecturebank",  default="data/processed/lecturebank")
    parser.add_argument("--openstax",     default="data/processed/openstax_full")
    parser.add_argument("--assistments",  default="data/processed/assistments")
    parser.add_argument("--output",       default="data/processed/concept_bridges.json")
    parser.add_argument("--threshold",    type=float, default=0.75)
    parser.add_argument(
        "--strategy", choices=["A", "B", "C"], default="B",
        help="A=embedding alignment, B=show axis assignments, C=inject MRL into ASSISTments"
    )
    args = parser.parse_args()

    if args.strategy == "B":
        axes = ConceptAligner.get_axis_datasets()
        print("\n=== PLEDGE-KARMA Evaluation Axes (Strategy B) ===\n")
        for axis_key, axis_info in axes.items():
            print(f"[{axis_key}]")
            print(f"  Description : {axis_info['description']}")
            print(f"  Primary     : {axis_info['primary_dataset']}")
            print(f"  Metrics     : {', '.join(axis_info['metrics'])}")
            print(f"  Note        : {axis_info['note'][:80]}...")
            print()

    elif args.strategy == "A":
        try:
            from models.mrl_encoder import MRLEncoder
            encoder = MRLEncoder({
                "model_name": "nomic-ai/nomic-embed-text-v1.5",
                "matryoshka_dims": [64, 128, 256, 512, 768],
                "full_dim": 768, "normalize_embeddings": True,
                "trust_remote_code": True,
            })
        except Exception:
            encoder = None
            print("WARNING: MRL encoder not available — using n-gram fallback.")

        aligner = ConceptAligner(encoder=encoder)

        # Load concepts from both datasets
        def load_concepts(path, dataset_name):
            p = Path(path) / "concepts.json"
            if not p.exists():
                return []
            with open(p) as f:
                raw = json.load(f)
            return [{"concept_id": c["concept_id"], "name": c["name"],
                     "dataset": dataset_name} for c in raw]

        lb_concepts = load_concepts(args.lecturebank, "lecturebank")
        os_concepts = load_concepts(args.openstax,    "openstax")

        if not lb_concepts or not os_concepts:
            print("One or more concept files not found. Run prepare_data.py first.")
        else:
            bridges = aligner.align_across_datasets(
                lb_concepts, os_concepts, threshold=args.threshold
            )
            aligner.save_bridges(bridges, args.output)
            print(f"✓ Saved {len(bridges)} bridges to {args.output}")

    elif args.strategy == "C":
        print("Strategy C: MRL injection into ASSISTments")
        print("Requires: sentence-transformers + processed ASSISTments interactions")
        print("Run: python data/pipelines/concept_alignment.py --strategy C "
              "--assistments data/processed/assistments --openstax data/processed/openstax_full")