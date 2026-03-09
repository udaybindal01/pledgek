"""
PLEDGE — Pedagogical Latent-state Estimation with
         Dependency-Graph-guided Evidence Retrieval

Core retrieval module implementing:
  1. Multi-scale FAISS index for MRL-based candidate retrieval
  2. Probabilistic admissibility filtering (using KARMA's dual state)
  3. Depth modulation function δ(Q, K_t)
  4. Submodular greedy set selection with cognitive load constraint
  5. Reactivation bonus for temporally decayed concepts

This module contains the formal retrieval objective:

  S* = argmax_{S: P(admissible|S,K_t) >= 1-δ}
         F(S|Q, K_t, δ(Q,K_t)) - λ_1·CL(S,K_t) + λ_3·R(S,K_t,t)

where F is student-state-conditioned submodular coverage function.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

from models.mrl_encoder import MRLEncoder, MRLEmbedding
from knowledge_graph.graph_builder import KnowledgeGraphBuilder, CorpusChunk
from karma.estimator import KARMAEstimator

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved corpus chunk with all associated scores."""
    chunk: CorpusChunk
    sim_64d: float
    sim_768d: float
    mrl_divergence: float
    admissibility_confidence: float
    depth_match_score: float
    cognitive_load_score: float
    reactivation_bonus: float
    submodular_marginal_gain: float
    final_score: float
    rank: int = 0


@dataclass
class RetrievalResult:
    """Complete result from a PLEDGE retrieval."""
    query: str
    retrieved_chunks: List[RetrievedChunk]
    student_depth_level: float          # δ(Q, K_t): computed depth target
    admissibility_violations: int       # Chunks filtered for admissibility
    reactivation_needed: List[str]      # Concept IDs needing reactivation
    metacognitive_profile: Dict
    metadata: Dict


class MultiScaleFAISSIndex:
    """
    FAISS index that supports retrieval at multiple MRL dimensions simultaneously.

    Maintains separate indices for each Matryoshka dimension to enable:
    1. Fast 64D retrieval for the initial candidate pool (cheap)
    2. 768D re-ranking for the final semantic check (expensive but on small set)
    """

    def __init__(self, dims: List[int], use_gpu: bool = False):
        self.dims = dims
        self.use_gpu = use_gpu
        self.indices: Dict[int, Any] = {}
        self.chunk_ids: List[str] = []        # Shared across all indices
        self._built = False

    def build(
        self,
        embeddings: List[MRLEmbedding],
        chunk_ids: List[str],
        index_type: str = "flat"
    ) -> None:
        """
        Build FAISS indices for all dimensions.

        Args:
            embeddings: List of MRL embeddings for all corpus chunks
            chunk_ids: Corresponding chunk IDs
            index_type: "flat" (exact, slow) or "ivf" (approximate, fast)
        """
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed. Install with: pip install faiss-cpu"
            )
        self.chunk_ids = chunk_ids
        n = len(embeddings)

        logger.info(f"Building FAISS indices for {n} chunks at dims {self.dims}")

        for dim in self.dims:
            vectors = np.stack([emb.at_dim(dim) for emb in embeddings]).astype(np.float32)

            if index_type == "flat" or n < 1000:
                index = faiss.IndexFlatIP(dim)  # Inner product = cosine (normalized)
            else:
                # IVF for large corpora
                n_clusters = min(int(np.sqrt(n)), 256)
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
                index.train(vectors)
                index.nprobe = min(n_clusters, 32)

            if self.use_gpu and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(index)

            index.add(vectors)
            self.indices[dim] = index

        self._built = True
        logger.info("FAISS indices built successfully")

    def search(
        self,
        query_emb: MRLEmbedding,
        dim: int,
        k: int
    ) -> List[Tuple[str, float]]:
        """
        Search at a specific dimension.

        Returns: List of (chunk_id, similarity_score) sorted by score desc
        """
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed. Install with: pip install faiss-cpu"
            )
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        query_vec = query_emb.at_dim(dim).reshape(1, -1).astype(np.float32)
        scores, indices = self.indices[dim].search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results

    def get_similarity(
        self,
        query_emb: MRLEmbedding,
        chunk_emb: MRLEmbedding,
        dim: int
    ) -> float:
        """Direct cosine similarity at a specific dimension."""
        q = query_emb.at_dim(dim)
        d = chunk_emb.at_dim(dim)
        return float(np.dot(q, d))


class CognitiveLoadEstimator:
    """
    Estimates the cognitive load of a text chunk for a specific student state.

    Based on Sweller's Cognitive Load Theory:
    - Intrinsic load: complexity of the content itself
    - Extraneous load: added by poor presentation (estimated from structure)
    - Germane load: productive load for schema formation

    We focus on intrinsic load, which is:
      CL(d, K_t) = Σ_{c in concepts(d)} novel_cost(c, K_t)
                 + dependency_depth_cost(d)

    This is student-specific: the same chunk has different CL for different students.
    """

    def __init__(self, config: Dict):
        self.novel_concept_cost = config.get("novel_concept_cost", 1.0)
        self.dependency_depth_cost = config.get("dependency_depth_cost", 0.5)
        self.working_memory_budget = config.get("working_memory_budget", 7.0)  # Miller's Law

    def compute(
        self,
        chunk: CorpusChunk,
        karma: KARMAEstimator,
        graph: KnowledgeGraphBuilder
    ) -> float:
        """
        Compute cognitive load of a chunk for a specific student.

        Args:
            chunk: The text chunk
            karma: Student's knowledge state estimator
            graph: Knowledge graph for dependency depth

        Returns:
            cognitive_load: float >= 0, normalized by working_memory_budget
        """
        total_load = 0.0

        for concept_id in chunk.concept_ids:
            p_obj, _, _ = karma.get_knowledge_state(concept_id)
            # Novel concepts (low mastery) cost more cognitive load
            novelty = 1.0 - p_obj
            total_load += self.novel_concept_cost * novelty

        # Dependency depth cost
        all_prereqs = set()
        for concept_id in chunk.concept_ids:
            prereqs = graph.get_prerequisites(concept_id, depth=2)
            all_prereqs.update(prereqs)

        total_load += self.dependency_depth_cost * len(all_prereqs)

        # Normalize by working memory budget
        normalized_load = total_load / self.working_memory_budget
        return float(normalized_load)

    def exceeds_budget(
        self,
        chunks: List[CorpusChunk],
        karma: KARMAEstimator,
        graph: KnowledgeGraphBuilder
    ) -> bool:
        """Check if a set of chunks together exceeds working memory budget."""
        total = sum(self.compute(c, karma, graph) for c in chunks)
        return total > 1.0  # Normalized budget = 1.0


class DepthModulator:
    """
    Computes the appropriate explanation depth for a given (query, K_t) pair.

    δ(Q, K_t) = Σ_{c ∈ related(Q)} p_t(c) · depth(c)
                ─────────────────────────────────────
                Σ_{c ∈ related(Q)} p_t(c)

    Depth levels:
        0 = introductory (intuitive explanations, minimal formalism)
        1 = intermediate (some formalism, assumes basic background)
        2 = advanced (full formalism, connects to research frontier)
    """

    def __init__(self, config: Dict):
        self.num_levels = config.get("num_levels", 3)
        self.depth_mismatch_penalty = config.get("depth_mismatch_penalty", 0.4)

    def compute_target_depth(
        self,
        related_concept_ids: List[str],
        karma: KARMAEstimator,
        graph: KnowledgeGraphBuilder
    ) -> float:
        """
        Compute the target explanation depth for the student's current state.

        Returns: float in [0, num_levels-1]
        """
        if not related_concept_ids:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for concept_id in related_concept_ids:
            p_obj, _, _ = karma.get_knowledge_state(concept_id)
            concept = graph.concepts.get(concept_id)
            if concept is None:
                continue
            depth = float(concept.depth_level)
            weighted_sum += p_obj * depth
            total_weight += p_obj

        if total_weight < 1e-9:
            return 0.0

        return weighted_sum / total_weight

    def compute_depth_match_score(
        self,
        chunk: CorpusChunk,
        target_depth: float
    ) -> float:
        """
        Score how well a chunk's depth matches the target depth.

        Returns: float in [0, 1], 1.0 = perfect match
        """
        depth_diff = abs(chunk.depth_level - target_depth)
        # Gaussian scoring: perfect match = 1.0, one level off = ~0.6
        score = np.exp(-0.5 * (depth_diff ** 2))
        return float(score)


class SubmodularRetriever:
    """
    Greedy submodular maximization for diverse, non-redundant retrieval.

    Objective function F(S|Q, K_t, δ) is monotone submodular under the
    assumption of diminishing returns — each additional document adds
    less new information than the previous one.

    Greedy algorithm achieves (1 - 1/e) ≈ 63% approximation of optimal set.

    The marginal gain function:
        Δ(d|S) = sim(q, d) · (1 - max_{d'∈S} overlap(d, d'))
    
    This naturally handles diversity without needing explicit clustering.
    """

    def __init__(self, config: Dict):
        self.diversity_weight = config.get("diversity_weight", 0.2)

    def compute_marginal_gain(
        self,
        candidate_emb: MRLEmbedding,
        query_emb: MRLEmbedding,
        selected_embs: List[MRLEmbedding],
        encoder: MRLEncoder,
        dim: int = 768
    ) -> float:
        """
        Compute marginal information gain of adding a candidate to current set S.

        Δ(d|S) = sim(q, d) · coverage_factor(d, S)
        where coverage_factor decreases as S already covers d's content.
        """
        # Base relevance to query
        relevance = encoder.compute_similarity(query_emb, candidate_emb, dim)

        if not selected_embs:
            return relevance

        # Compute max similarity to already-selected documents
        max_overlap = max(
            encoder.compute_similarity(candidate_emb, sel_emb, dim)
            for sel_emb in selected_embs
        )

        # Coverage factor: 1.0 if no overlap, approaches 0 if fully covered
        coverage_factor = 1.0 - self.diversity_weight * max_overlap

        return float(relevance * coverage_factor)

    def greedy_select(
        self,
        candidates: List[Tuple[str, MRLEmbedding, float]],  # (chunk_id, emb, base_score)
        query_emb: MRLEmbedding,
        encoder: MRLEncoder,
        k: int,
        dim: int = 768
    ) -> List[Tuple[str, float]]:
        """
        Greedy submodular set selection.

        Returns: List of (chunk_id, marginal_gain) for top-k selected chunks
        """
        selected = []
        selected_embs = []
        selected_with_gains = []
        remaining = list(candidates)

        for _ in range(min(k, len(remaining))):
            if not remaining:
                break

            # Compute marginal gain for all remaining candidates
            gains = []
            for chunk_id, emb, base_score in remaining:
                mg = self.compute_marginal_gain(
                    emb, query_emb, selected_embs, encoder, dim
                )
                # Weight marginal gain by base score (admissibility, depth, etc.)
                combined = 0.6 * mg + 0.4 * base_score
                gains.append((chunk_id, emb, base_score, combined))

            # Select candidate with highest combined gain
            gains.sort(key=lambda x: x[3], reverse=True)
            best_id, best_emb, best_base, best_gain = gains[0]

            selected.append(best_id)
            selected_embs.append(best_emb)
            selected_with_gains.append((best_id, best_gain))

            # Remove from remaining
            remaining = [(cid, emb, score) for cid, emb, score in remaining
                        if cid != best_id]

        return selected_with_gains


class PLEDGERetriever:
    """
    PLEDGE: Full Pedagogical Retrieval Engine.

    Orchestrates the complete retrieval pipeline:
    1. 64D fast retrieval → candidate pool
    2. Admissibility filtering (using KARMA)
    3. Depth modulation scoring
    4. Cognitive load scoring
    5. Reactivation bonus computation
    6. Submodular greedy selection
    7. 768D final re-ranking
    """

    def __init__(
        self,
        config: Dict,
        encoder: MRLEncoder,
        graph: KnowledgeGraphBuilder,
        karma: KARMAEstimator
    ):
        self.config = config
        self.encoder = encoder
        self.graph = graph
        self.karma = karma

        # Sub-components
        retrieval_config = config.get("retrieval", {})
        self.candidate_pool_size = retrieval_config.get("candidate_pool_size", 100)
        self.final_k = retrieval_config.get("final_k", 10)
        self.lambda_cl = retrieval_config.get("lambda_cognitive_load", 0.3)
        self.lambda_reactivation = retrieval_config.get("lambda_reactivation", 0.4)

        admissibility_config = config.get("admissibility", {})
        self.admissibility_threshold = admissibility_config.get("confidence_threshold", 0.80)
        self.hard_constraint = admissibility_config.get("hard_constraint", False)
        self.uncertainty_penalty = admissibility_config.get("uncertainty_penalty", 0.3)

        self.depth_modulator = DepthModulator(config.get("depth", {}))
        self.cl_estimator = CognitiveLoadEstimator(config.get("cognitive_load", {}))
        self.submodular = SubmodularRetriever(retrieval_config)

        # FAISS index
        self.faiss_index = MultiScaleFAISSIndex(
            dims=encoder.dims,
            use_gpu="cuda" in config.get("device", "cpu")
        )
        self._index_built = False
        self._chunk_embeddings: Dict[str, MRLEmbedding] = {}

    def build_index(
        self,
        chunks: List[CorpusChunk],
        show_progress: bool = True
    ) -> None:
        """
        Build the FAISS index from corpus chunks.
        Must be called before retrieve().
        """
        logger.info(f"Building index for {len(chunks)} chunks...")

        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        embeddings = self.encoder.encode_documents_batch(texts, show_progress=show_progress)

        # Store embeddings for re-ranking
        for chunk_id, emb in zip(chunk_ids, embeddings):
            self._chunk_embeddings[chunk_id] = emb

        self.faiss_index.build(embeddings, chunk_ids)
        self._index_built = True
        logger.info("Index built successfully")

    def retrieve(
        self,
        query: str,
        student_concept_history: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        Main retrieval function. Full PLEDGE pipeline.

        Args:
            query: Student's question/query
            student_concept_history: Recently engaged concept IDs (for ZPD)

        Returns:
            RetrievalResult with ranked chunks and all diagnostic information
        """
        if not self._index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        # ─── Step 1: Encode query at all MRL scales ───
        query_emb = self.encoder.encode_query(query)

        # ─── Step 2: Identify query-related concepts ───
        related_concepts = self._identify_query_concepts(query_emb)

        # ─── Step 3: Compute target depth δ(Q, K_t) ───
        target_depth = self.depth_modulator.compute_target_depth(
            related_concepts,
            self.karma,
            self.graph
        )

        # ─── Step 4: 64D fast retrieval → candidate pool ───
        candidates_64d = self.faiss_index.search(
            query_emb, dim=64, k=self.candidate_pool_size
        )
        candidate_chunk_ids = [cid for cid, _ in candidates_64d]

        # ─── Step 5: Score and filter candidates ───
        scored_candidates = []
        n_admissibility_violations = 0
        reactivation_needed = set()

        for chunk_id, sim_64d in candidates_64d:
            chunk = self.graph.chunks.get(chunk_id)
            if chunk is None:
                continue

            chunk_emb = self._chunk_embeddings.get(chunk_id)
            if chunk_emb is None:
                continue

            # 768D similarity
            sim_768d = self.encoder.compute_similarity(query_emb, chunk_emb, 768)

            # MRL divergence (metacognitive signal)
            mrl_div = self.encoder.compute_dimensional_divergence(
                query_emb, chunk_emb, coarse_dim=64, fine_dim=768
            )

            # ─ Admissibility check ─
            admissibility_conf = self.karma.compute_admissibility_confidence(
                chunk.prerequisite_concept_ids,
                delta_penalty=self.uncertainty_penalty
            )

            if self.hard_constraint and admissibility_conf < self.admissibility_threshold:
                n_admissibility_violations += 1
                continue

            # ─ Check reactivation needs ─
            reactivation_bonus = 0.0
            for concept_id in chunk.concept_ids:
                if self.karma.needs_reactivation(concept_id):
                    reactivation_needed.add(concept_id)
                    # Bonus for chunks that can help reactivate decayed concepts
                    if chunk.depth_level == 0:  # Introductory = good for reactivation
                        reactivation_bonus += 0.1

            # ─ Depth match score ─
            depth_score = self.depth_modulator.compute_depth_match_score(
                chunk, target_depth
            )

            # ─ Cognitive load score (lower is better) ─
            cl_score = self.cl_estimator.compute(chunk, self.karma, self.graph)

            # ─ Combined base score ─
            base_score = (
                0.4 * sim_768d
                + 0.2 * sim_64d
                + 0.2 * depth_score
                + self.lambda_reactivation * reactivation_bonus
                - self.lambda_cl * cl_score
            )

            # Apply soft admissibility penalty if not hard constraint
            if not self.hard_constraint:
                if admissibility_conf < self.admissibility_threshold:
                    n_admissibility_violations += 1
                    admissibility_weight = admissibility_conf / self.admissibility_threshold
                    base_score *= admissibility_weight

            scored_candidates.append((
                chunk_id, chunk_emb, base_score,
                sim_64d, sim_768d, mrl_div,
                admissibility_conf, depth_score, cl_score, reactivation_bonus
            ))

        if not scored_candidates:
            logger.warning("No admissible candidates found. Relaxing constraint.")
            return self._fallback_retrieve(query, query_emb, target_depth)

        # ─── Step 6: Submodular greedy selection ───
        submodular_input = [
            (cid, emb, score)
            for cid, emb, score, *_ in scored_candidates
        ]
        selected = self.submodular.greedy_select(
            submodular_input,
            query_emb,
            self.encoder,
            k=self.final_k,
            dim=768
        )
        selected_ids = {cid for cid, _ in selected}

        # ─── Step 7: Assemble final result ───
        scored_dict = {
            cid: (s, s64, s768, div, adm, dep, cl, react)
            for cid, _, s, s64, s768, div, adm, dep, cl, react in scored_candidates
        }

        final_chunks = []
        for rank, (chunk_id, mg) in enumerate(selected):
            chunk = self.graph.chunks[chunk_id]
            s, s64, s768, div, adm, dep, cl, react = scored_dict[chunk_id]

            final_chunks.append(RetrievedChunk(
                chunk=chunk,
                sim_64d=s64,
                sim_768d=s768,
                mrl_divergence=div,
                admissibility_confidence=adm,
                depth_match_score=dep,
                cognitive_load_score=cl,
                reactivation_bonus=react,
                submodular_marginal_gain=mg,
                final_score=s,
                rank=rank
            ))

        metacognitive_profile = self.karma.get_metacognitive_profile()

        return RetrievalResult(
            query=query,
            retrieved_chunks=final_chunks,
            student_depth_level=target_depth,
            admissibility_violations=n_admissibility_violations,
            reactivation_needed=list(reactivation_needed),
            metacognitive_profile=metacognitive_profile,
            metadata={
                "candidate_pool_size": len(candidates_64d),
                "post_admissibility_size": len(scored_candidates),
                "target_depth": target_depth,
                "mrl_dims_used": self.encoder.dims
            }
        )

    def _identify_query_concepts(
        self,
        query_emb: MRLEmbedding,
        top_k: int = 5
    ) -> List[str]:
        """
        Identify which concepts a query is most related to.
        Used for computing target depth and ZPD routing.
        """
        if not self.graph.concepts:
            return []

        concept_sims = []
        for concept_id, concept in self.graph.concepts.items():
            if concept.embedding is None:
                continue
            concept_emb = MRLEmbedding(
                full_embedding=concept.embedding,
                dims=self.encoder.dims
            )
            sim = self.encoder.compute_similarity(query_emb, concept_emb, 256)
            concept_sims.append((concept_id, sim))

        concept_sims.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in concept_sims[:top_k]]

    def _fallback_retrieve(
        self,
        query: str,
        query_emb: MRLEmbedding,
        target_depth: float
    ) -> RetrievalResult:
        """Fallback to standard top-k 768D retrieval when no admissible candidates."""
        logger.warning("Using fallback retrieval (no admissibility constraint)")
        results_768d = self.faiss_index.search(query_emb, dim=768, k=self.final_k)

        final_chunks = []
        for rank, (chunk_id, sim_768d) in enumerate(results_768d):
            chunk = self.graph.chunks.get(chunk_id)
            if chunk is None:
                continue
            chunk_emb = self._chunk_embeddings.get(chunk_id)
            sim_64d = (self.encoder.compute_similarity(query_emb, chunk_emb, 64)
                      if chunk_emb else 0.0)

            final_chunks.append(RetrievedChunk(
                chunk=chunk,
                sim_64d=sim_64d,
                sim_768d=sim_768d,
                mrl_divergence=0.0,
                admissibility_confidence=0.5,
                depth_match_score=0.5,
                cognitive_load_score=0.5,
                reactivation_bonus=0.0,
                submodular_marginal_gain=sim_768d,
                final_score=sim_768d,
                rank=rank
            ))

        return RetrievalResult(
            query=query,
            retrieved_chunks=final_chunks,
            student_depth_level=target_depth,
            admissibility_violations=0,
            reactivation_needed=[],
            metacognitive_profile={},
            metadata={"fallback": True}
        )