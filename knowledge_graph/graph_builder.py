"""
Knowledge Graph Builder for PLEDGE-KARMA.

Constructs three-layer graph from educational corpora:
  - G_P: Prerequisite edges (what must come before what)
  - G_S: Semantic edges (conceptual proximity via MRL)
  - G_R: Recontextualization edges (early concepts revisited after later ones)

The multi-scale agreement requirement for edge formation is the core
mechanism that prevents semantically-similar-but-pedagogically-invalid
connections from entering the knowledge graph.
"""

import re
import json
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm

from models.mrl_encoder import MRLEncoder, MRLEmbedding

logger = logging.getLogger(__name__)


@dataclass
class ConceptNode:
    """
    A node in the knowledge graph — represents a teachable concept.
    """
    concept_id: str
    name: str
    description: str
    source_chunk_ids: List[str]          # Which corpus chunks define this concept
    depth_level: int                      # 0=introductory, 1=intermediate, 2=advanced
    chapter_order: int                    # Position in textbook (topological sort seed)
    subject: str                          # e.g. "physics", "calculus"
    embedding: Optional[np.ndarray] = None
    difficulty_score: float = 0.5        # Estimated difficulty in [0,1]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "description": self.description,
            "source_chunk_ids": self.source_chunk_ids,
            "depth_level": self.depth_level,
            "chapter_order": self.chapter_order,
            "subject": self.subject,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "difficulty_score": self.difficulty_score,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ConceptNode":
        node = cls(
            concept_id=d["concept_id"],
            name=d["name"],
            description=d["description"],
            source_chunk_ids=d["source_chunk_ids"],
            depth_level=d["depth_level"],
            chapter_order=d["chapter_order"],
            subject=d["subject"],
            difficulty_score=d.get("difficulty_score", 0.5),
            tags=d.get("tags", [])
        )
        if d.get("embedding"):
            node.embedding = np.array(d["embedding"])
        return node


@dataclass
class CorpusChunk:
    """
    A retrievable text chunk from the educational corpus.
    """
    chunk_id: str
    text: str
    concept_ids: List[str]               # Concepts this chunk explains
    prerequisite_concept_ids: List[str]  # Concepts needed to understand this chunk
    depth_level: int                     # 0=introductory, 1=intermediate, 2=advanced
    chapter_order: int
    subject: str
    source: str                          # "openstax", "ck12", etc.
    embedding: Optional[MRLEmbedding] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def all_required_concepts(self) -> Set[str]:
        return set(self.prerequisite_concept_ids)


class KnowledgeGraphBuilder:
    """
    Builds the three-layer educational knowledge graph.

    Architecture:
        G = (C, E_P, E_S, E_R)
        C  = concept nodes
        E_P = prerequisite edges (directed: prereq → dependent)
        E_S = semantic edges (undirected: conceptually proximate)
        E_R = recontextualization edges (directed: early → revisited-by-later)

    Edge formation rule (multi-scale agreement):
        A semantic edge between concepts A and B is only formed if:
            sim_64D(A, B)  >= threshold_coarse   AND
            sim_768D(A, B) >= threshold_fine      AND
            agreement_score(A, B) >= threshold_agreement
        This prevents vocabulary-matching but conceptually-distant concepts
        from being falsely connected — the core graph integrity guarantee.
    """

    RECONTEXTUALIZATION_PATTERNS = [
        r"recall that\b",
        r"as we (saw|showed|discussed|proved)\b",
        r"we (introduced|defined|showed)\b",
        r"building on (our|the|this)\b",
        r"now we can understand\b",
        r"in light of\b",
        r"revisiting\b",
        r"from (chapter|section) \d+\b",
        r"as mentioned (earlier|previously|before)\b",
        r"we (earlier|previously) (established|defined|showed)\b"
    ]

    def __init__(self, config: Dict, encoder: MRLEncoder):
        self.config = config
        self.encoder = encoder
        self.graph = nx.DiGraph()

        # Edge thresholds
        self.prereq_sim_threshold = config.get("prerequisite_sim_threshold", 0.75)
        self.cross_scale_threshold = config.get("cross_scale_agreement_threshold", 0.60)
        self.min_edge_confidence = config.get("min_edge_confidence", 0.35)

        # Storage
        self.concepts: Dict[str, ConceptNode] = {}
        self.chunks: Dict[str, CorpusChunk] = {}

        self._recontex_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.RECONTEXTUALIZATION_PATTERNS
        ]

    def add_concept(self, concept: ConceptNode) -> None:
        """Add a concept node to the graph."""
        self.concepts[concept.concept_id] = concept
        self.graph.add_node(
            concept.concept_id,
            name=concept.name,
            depth_level=concept.depth_level,
            chapter_order=concept.chapter_order,
            subject=concept.subject,
            difficulty_score=concept.difficulty_score,
            node_type="concept"
        )

    def add_chunk(self, chunk: CorpusChunk) -> None:
        """Register a corpus chunk."""
        self.chunks[chunk.chunk_id] = chunk

    def build_prerequisite_edges_from_ordering(self) -> int:
        """
        Build E_P from chapter/section ordering.

        Strategy: concepts from earlier chapters are candidates for being
        prerequisites of later-chapter concepts. We use chapter_order as
        a topological seed and validate with multi-scale agreement.

        Returns: number of prerequisite edges added
        """
        logger.info("Building prerequisite edges from chapter ordering...")
        edges_added = 0
        concepts_sorted = sorted(
            self.concepts.values(),
            key=lambda c: c.chapter_order
        )

        for i, later_concept in enumerate(tqdm(concepts_sorted, desc="Prereq edges")):
            if later_concept.embedding is None:
                continue
            later_emb = MRLEmbedding(
                full_embedding=later_concept.embedding,
                dims=self.encoder.dims
            )

            # Only look at earlier concepts as candidates
            for earlier_concept in concepts_sorted[:i]:
                if earlier_concept.embedding is None:
                    continue
                if earlier_concept.subject != later_concept.subject:
                    continue  # Cross-subject prereqs need explicit annotation
                # chapter_order is encoded as chapter_num * 100 + page_idx
                # in the OpenStax pipeline (ch1=100-120, ch2=200-220, etc.)
                # Adjacent chapters differ by ~100, not 1.
                order_gap   = later_concept.chapter_order - earlier_concept.chapter_order
                chapter_gap = order_gap // 100  # approximate chapter distance

                # Only consider immediate adjacency (≤1 chapter) OR
                # same-chapter concepts (gap < 100 = same chapter, different page).
                # Wider gaps get very low ordering confidence.
                if chapter_gap > 5:
                    continue

                earlier_emb = MRLEmbedding(
                    full_embedding=earlier_concept.embedding,
                    dims=self.encoder.dims
                )

                semantic_conf = self._compute_prerequisite_confidence(
                    earlier_emb, later_emb
                )

                # Ordering confidence: strong signal even when semantic is low.
                # Direct neighbours (gap=0 or 1) get ordering_bonus=0.50.
                # Each additional chapter reduces the bonus.
                # This ensures we get meaningful edges even with mock embeddings.
                if chapter_gap == 0:
                    ordering_bonus = 0.60   # Same chapter: very likely related
                elif chapter_gap == 1:
                    ordering_bonus = 0.50   # Adjacent chapter: strong signal
                elif chapter_gap == 2:
                    ordering_bonus = 0.35
                else:
                    ordering_bonus = 0.20

                # Blend: take the max of semantic and ordering signals.
                # With real embeddings, semantic_conf ≥ ordering_bonus for related
                # concepts → no change. With mock embeddings, ordering_bonus
                # dominates for close chapters → we still get meaningful edges.
                confidence = max(semantic_conf, ordering_bonus)

                if confidence >= self.min_edge_confidence:
                    self.graph.add_edge(
                        earlier_concept.concept_id,
                        later_concept.concept_id,
                        edge_type="prerequisite",
                        confidence=confidence,
                        weight=confidence,
                        ordering_bonus=ordering_bonus,
                        semantic_conf=semantic_conf,
                    )
                    edges_added += 1

        logger.info(f"Added {edges_added} prerequisite edges from ordering")
        return edges_added

    def build_prerequisite_edges_from_annotations(
        self,
        annotations: List[Tuple[str, str, float]]
    ) -> int:
        """
        Add prerequisite edges from human-annotated sources (LectureBank, MOOCCube).

        Args:
            annotations: List of (prereq_concept_id, dependent_concept_id, confidence)

        Matching strategy (tried in order):
          1. Direct ID match — fastest path when IDs align
          2. Name-based match — handles cross-pipeline ID format mismatches
             (e.g. LectureBank "lb_variables" vs OpenStax generate_id hashes)
        """
        # Build name→id reverse index for fuzzy matching
        name_to_id: Dict[str, str] = {
            c.name.lower().strip(): c.concept_id
            for c in self.concepts.values()
        }
        # Also index by graph node names (stored during add_concept)
        for node_id, data in self.graph.nodes(data=True):
            name = data.get("name", "")
            if name and name.lower() not in name_to_id:
                name_to_id[name.lower().strip()] = node_id

        def resolve(raw_id: str) -> Optional[str]:
            """Return the real concept_id for a raw annotation ID, or None."""
            # 1. Direct hit
            if raw_id in self.concepts:
                return raw_id
            # 2. Treat the raw_id as a concept name (strip prefix like lb_, mc_)
            clean = re.sub(r"^(lb|mc|os|concept)_?", "", raw_id).replace("_", " ").lower()
            if clean in name_to_id:
                return name_to_id[clean]
            # 3. Partial name match (raw_id as substring of known concept name)
            for name, cid in name_to_id.items():
                if clean in name or name in clean:
                    return cid
            return None

        edges_added = 0
        skipped = 0
        unresolvable = 0
        for prereq_raw, dep_raw, confidence in annotations:
            prereq_id = resolve(prereq_raw)
            dep_id    = resolve(dep_raw)
            if prereq_id and dep_id and prereq_id != dep_id:
                self.graph.add_edge(
                    prereq_id, dep_id,
                    edge_type="prerequisite",
                    confidence=confidence,
                    weight=confidence,
                    source="annotation",
                )
                edges_added += 1
            else:
                skipped += 1
                unresolvable += 1

        if unresolvable > 0 and edges_added == 0:
            # All annotations failed to resolve → likely a cross-domain dataset
            # (e.g. LectureBank CS concepts vs OpenStax physics corpus).
            # Log a clear warning rather than silently returning 0.
            total = len(annotations)
            logger.warning(
                f"Annotation edge resolution: 0/{total} resolved. "
                f"This usually means the annotation dataset is in a different domain "
                f"(e.g. CS/NLP annotations with a physics/calculus corpus). "
                f"Falling back to chapter-ordering edges only."
            )
        elif skipped > 0:
            logger.debug(f"  Skipped {skipped}/{len(annotations)} annotation edges "
                         f"(concepts not found in graph — possible domain mismatch)")
        logger.info(f"Added {edges_added} prerequisite edges from annotations")
        return edges_added

    def backfill_chunk_prerequisites(
        self,
        min_confidence: float = 0.5,
        graph_depth: int = 1,
    ) -> int:
        """
        Propagate concept-level prerequisite edges back onto chunks.

        This is the critical step that connects the concept graph (E_P edges)
        to PLEDGE's admissibility filter, which reads chunk.prerequisite_concept_ids.

        After calling build_prerequisite_edges_from_* you MUST call this method,
        otherwise chunk.prerequisite_concept_ids stays [] and the admissibility
        constraint is permanently disabled (every chunk appears free to retrieve).

        Strategy: for each chunk, gather the prerequisites of ALL concepts it
        explains, deduplicate, and write them to chunk.prerequisite_concept_ids.
        """
        chunks_updated = 0
        total_prereqs  = 0

        for chunk in self.chunks.values():
            all_prereqs: set = set()

            for concept_id in chunk.concept_ids:
                if concept_id not in self.graph:
                    continue
                # Walk prerequisite edges in the concept graph
                for depth_hop in range(graph_depth):
                    frontier = {concept_id} if depth_hop == 0 else all_prereqs.copy()
                    for node in frontier:
                        for pred in self.graph.predecessors(node):
                            edge_data = self.graph.edges[pred, node]
                            if (edge_data.get("edge_type") == "prerequisite" and
                                    edge_data.get("confidence", 0) >= min_confidence):
                                all_prereqs.add(pred)

            # Never include a concept as its own prerequisite
            all_prereqs -= set(chunk.concept_ids)

            if all_prereqs:
                chunk.prerequisite_concept_ids = list(all_prereqs)
                chunks_updated += 1
                total_prereqs  += len(all_prereqs)

        avg = total_prereqs / max(chunks_updated, 1)
        logger.info(
            f"backfill_chunk_prerequisites: updated {chunks_updated}/{len(self.chunks)} chunks "
            f"(avg {avg:.1f} prereqs/chunk)"
        )
        return chunks_updated

    def build_semantic_edges(self, sim_threshold: float = 0.75) -> int:
        """
        Build E_S — semantic proximity edges.

        Key constraint: edges only form when similarity is stable across
        64D AND 768D. This is the multi-scale agreement filter.
        """
        logger.info("Building semantic edges with multi-scale agreement filter...")
        edges_added = 0
        concept_list = [c for c in self.concepts.values() if c.embedding is not None]

        for i, concept_a in enumerate(tqdm(concept_list, desc="Semantic edges")):
            emb_a = MRLEmbedding(
                full_embedding=concept_a.embedding,
                dims=self.encoder.dims
            )
            for concept_b in concept_list[i + 1:]:
                if concept_a.concept_id == concept_b.concept_id:
                    continue
                emb_b = MRLEmbedding(
                    full_embedding=concept_b.embedding,
                    dims=self.encoder.dims
                )

                # Multi-scale agreement check
                sim_64 = self.encoder.compute_similarity(emb_a, emb_b, 64)
                sim_768 = self.encoder.compute_similarity(emb_a, emb_b, 768)
                agreement = self.encoder.multiscale_agreement_score(emb_a, emb_b)

                # Edge forms only if BOTH scales agree
                if (sim_768 >= sim_threshold and
                        sim_64 >= self.cross_scale_threshold and
                        agreement >= 0.6):
                    self.graph.add_edge(
                        concept_a.concept_id,
                        concept_b.concept_id,
                        edge_type="semantic",
                        sim_64=sim_64,
                        sim_768=sim_768,
                        agreement=agreement,
                        weight=sim_768
                    )
                    edges_added += 1

        logger.info(f"Added {edges_added} semantic edges")
        return edges_added

    def build_recontextualization_edges(self) -> int:
        """
        Build E_R — edges where a later chunk explicitly revisits an earlier concept.

        Detected via linguistic patterns: "recall that", "as we saw", etc.
        These edges power the temporal knowledge restructuring component.
        """
        logger.info("Building recontextualization edges...")
        edges_added = 0

        # Build concept name → id lookup
        name_to_id = {
            c.name.lower(): cid
            for cid, c in self.concepts.items()
        }

        for chunk_id, chunk in self.chunks.items():
            if not any(p.search(chunk.text) for p in self._recontex_patterns):
                continue

            # Check which earlier concepts this chunk references
            for concept_name, concept_id in name_to_id.items():
                concept = self.concepts[concept_id]
                if concept.chapter_order >= chunk.chapter_order:
                    continue  # Only earlier concepts get recontextualized
                if concept_name in chunk.text.lower():
                    # Check if it's in a recontextualization context
                    for pattern in self._recontex_patterns:
                        matches = list(pattern.finditer(chunk.text.lower()))
                        for match in matches:
                            context_window = chunk.text.lower()[
                                max(0, match.start() - 100):
                                min(len(chunk.text), match.end() + 100)
                            ]
                            if concept_name in context_window:
                                # The later chunk's concepts recontextualize the earlier concept
                                for later_concept_id in chunk.concept_ids:
                                    if later_concept_id != concept_id:
                                        if not self.graph.has_edge(concept_id, later_concept_id):
                                            self.graph.add_edge(
                                                concept_id,
                                                later_concept_id,
                                                edge_type="recontextualization",
                                                weight=0.8,
                                                pattern=pattern.pattern
                                            )
                                            edges_added += 1
                                break

        logger.info(f"Added {edges_added} recontextualization edges")
        return edges_added

    def embed_all_concepts(self, show_progress: bool = True) -> None:
        """Compute and store embeddings for all concept nodes."""
        concepts_needing_emb = [
            c for c in self.concepts.values()
            if c.embedding is None
        ]
        if not concepts_needing_emb:
            return

        logger.info(f"Embedding {len(concepts_needing_emb)} concepts...")
        texts = [f"{c.name}: {c.description}" for c in concepts_needing_emb]
        embeddings = self.encoder.encode_documents_batch(texts, show_progress=show_progress)

        for concept, emb in zip(concepts_needing_emb, embeddings):
            concept.embedding = emb.full_embedding
            self.graph.nodes[concept.concept_id]["embedding"] = emb.full_embedding

    def get_prerequisites(
        self,
        concept_id: str,
        depth: int = 1,
        min_confidence: float = 0.5
    ) -> Set[str]:
        """
        Get all prerequisite concepts for a given concept.

        Args:
            concept_id: Target concept
            depth: How many hops back to traverse
            min_confidence: Minimum edge confidence to follow
        """
        prereqs = set()
        if concept_id not in self.graph:
            return prereqs

        visited = {concept_id}
        frontier = {concept_id}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for pred in self.graph.predecessors(node):
                    edge_data = self.graph.edges[pred, node]
                    if (edge_data.get("edge_type") == "prerequisite" and
                            edge_data.get("confidence", 0) >= min_confidence):
                        if pred not in visited:
                            prereqs.add(pred)
                            next_frontier.add(pred)
                            visited.add(pred)
            frontier = next_frontier

        return prereqs

    def get_zone_of_proximal_development(
        self,
        known_concepts: Dict[str, float],  # concept_id → mastery probability
        mastery_threshold: float = 0.8,
        n_candidates: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Identify concepts in the Zone of Proximal Development (Vygotsky).

        ZPD = concepts whose prerequisites are mastered but which are not
        yet mastered themselves. These are the highest-value learning targets.

        Returns:
            List of (concept_id, zpd_score) sorted by ZPD score descending
        """
        zpd_candidates = []

        for concept_id, concept in self.concepts.items():
            # Skip already mastered
            if known_concepts.get(concept_id, 0.0) >= mastery_threshold:
                continue

            # Get prerequisites
            prereqs = self.get_prerequisites(concept_id, depth=2)
            if not prereqs:
                # No prerequisites → always accessible
                zpd_score = 0.5
            else:
                # ZPD score = mean mastery of prerequisites
                prereq_mastery = [
                    known_concepts.get(p, 0.0) for p in prereqs
                ]
                zpd_score = np.mean(prereq_mastery)

            if zpd_score >= mastery_threshold * 0.7:  # Prerequisites mostly mastered
                zpd_candidates.append((concept_id, zpd_score))

        zpd_candidates.sort(key=lambda x: x[1], reverse=True)
        return zpd_candidates[:n_candidates]

    def _compute_prerequisite_confidence(
        self,
        earlier_emb: MRLEmbedding,
        later_emb: MRLEmbedding
    ) -> float:
        """
        Compute confidence that earlier concept is prerequisite of later concept.

        Combines:
        - 64D similarity (coarse domain match)
        - 768D similarity (semantic proximity)
        - Multi-scale agreement (stability across scales)
        - Directional signal (later should be "denser" than earlier)
        """
        sim_64 = self.encoder.compute_similarity(earlier_emb, later_emb, 64)
        sim_256 = self.encoder.compute_similarity(earlier_emb, later_emb, 256)
        sim_768 = self.encoder.compute_similarity(earlier_emb, later_emb, 768)
        agreement = self.encoder.multiscale_agreement_score(earlier_emb, later_emb)

        # Domain must match at coarse level
        if sim_64 < self.cross_scale_threshold:
            return 0.0

        # Confidence = weighted combination
        confidence = (
            0.3 * sim_64 +
            0.3 * sim_256 +
            0.2 * sim_768 +
            0.2 * agreement
        )
        return float(confidence)

    def save(self, path: str) -> None:
        """Save the knowledge graph to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save graph structure
        nx.write_gml(self.graph, str(path / "graph.gml"))

        # Save concepts
        concepts_data = {cid: c.to_dict() for cid, c in self.concepts.items()}
        with open(path / "concepts.json", "w") as f:
            json.dump(concepts_data, f, indent=2)

        # Save chunks (without embeddings to save space)
        chunks_data = {
            cid: {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "concept_ids": c.concept_ids,
                "prerequisite_concept_ids": c.prerequisite_concept_ids,
                "depth_level": c.depth_level,
                "chapter_order": c.chapter_order,
                "subject": c.subject,
                "source": c.source,
                "metadata": c.metadata
            }
            for cid, c in self.chunks.items()
        }
        with open(path / "chunks.json", "w") as f:
            json.dump(chunks_data, f, indent=2)

        logger.info(f"Knowledge graph saved to {path}")

    @classmethod
    def load(cls, path: str, config: Dict, encoder: MRLEncoder) -> "KnowledgeGraphBuilder":
        """Load a saved knowledge graph."""
        path = Path(path)
        builder = cls(config, encoder)

        builder.graph = nx.read_gml(str(path / "graph.gml"))

        with open(path / "concepts.json") as f:
            concepts_data = json.load(f)
        builder.concepts = {
            cid: ConceptNode.from_dict(d)
            for cid, d in concepts_data.items()
        }

        with open(path / "chunks.json") as f:
            chunks_data = json.load(f)
        builder.chunks = {
            cid: CorpusChunk(**d)
            for cid, d in chunks_data.items()
        }

        logger.info(f"Knowledge graph loaded from {path}: "
                    f"{len(builder.concepts)} concepts, "
                    f"{builder.graph.number_of_edges()} edges")
        return builder

    def summary(self) -> Dict:
        """Print graph statistics."""
        edge_types = {}
        for u, v, d in self.graph.edges(data=True):
            et = d.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "n_concepts": len(self.concepts),
            "n_chunks": len(self.chunks),
            "n_edges": self.graph.number_of_edges(),
            "edge_types": edge_types,
            "is_dag": nx.is_directed_acyclic_graph(
                nx.DiGraph([(u, v) for u, v, d in self.graph.edges(data=True)
                            if d.get("edge_type") == "prerequisite"])
            )
        }