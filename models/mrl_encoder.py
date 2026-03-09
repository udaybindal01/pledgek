"""
MRL Encoder — Matryoshka Representation Learning Encoder
Uses Nomic-embed-text-v1.5 which natively supports MRL with nested dimensions.

Key capability: encode documents/queries at multiple granularities simultaneously,
enabling the dimensional divergence signal used in KARMA's metacognitive gap estimation.
"""

try:
    import torch  # noqa: F401  — only needed by sentence_transformers
except ImportError:
    torch = None  # type: ignore

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MRLEmbedding:
    """
    Container for a multi-scale MRL embedding.
    Stores the full embedding and provides lazy slicing at any registered dimension.
    """
    full_embedding: np.ndarray          # Shape: (full_dim,)
    dims: List[int]                     # Registered Matryoshka dims e.g. [64,128,256,512,768]
    text: Optional[str] = None

    def at_dim(self, dim: int) -> np.ndarray:
        """Return normalized embedding truncated to `dim` dimensions."""
        if dim not in self.dims:
            raise ValueError(f"Dim {dim} not in registered dims {self.dims}")
        truncated = self.full_embedding[:dim].copy()
        norm = np.linalg.norm(truncated)
        return truncated / (norm + 1e-9)

    def dimensional_divergence(self, dim_coarse: int, dim_fine: int) -> float:
        """
        Compute self-divergence between two scales.
        Used internally to measure embedding stability.
        """
        e_coarse = self.at_dim(dim_coarse)
        e_fine = self.at_dim(dim_fine)
        e_coarse_padded = np.zeros(dim_fine)
        e_coarse_padded[:dim_coarse] = e_coarse
        return float(np.dot(e_coarse_padded, e_fine))

    def to_dict(self) -> Dict:
        return {
            "full_embedding": self.full_embedding.tolist(),
            "dims": self.dims,
            "text": self.text
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MRLEmbedding":
        return cls(
            full_embedding=np.array(d["full_embedding"]),
            dims=d["dims"],
            text=d.get("text")
        )


class MRLEncoder:
    """
    Matryoshka Representation Learning Encoder.

    Wraps Nomic-embed-text-v1.5 to produce nested embeddings at multiple
    dimensionalities. Critical for PLEDGE-KARMA because:

    1. 64D embeddings  → coarse domain/prerequisite-level signals
    2. 768D embeddings → fine-grained semantic matching
    3. Dimensional divergence → metacognitive gap proxy (KARMA)
    4. Multi-scale agreement → prerequisite edge validation (PLEDGE graph)
    """

    MATRYOSHKA_DIMS = [64, 128, 256, 512, 768]

    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get("model_name", "nomic-ai/nomic-embed-text-v1.5")
        self.dims = config.get("matryoshka_dims", self.MATRYOSHKA_DIMS)
        self.full_dim = config.get("full_dim", 768)
        self.batch_size = config.get("batch_size", 32)
        self.device = config.get("device", "cpu")
        self.normalize = config.get("normalize_embeddings", True)
        self.max_seq_length = config.get("max_seq_length", 512)

        logger.info(f"Loading MRL encoder: {self.model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=config.get("trust_remote_code", True),
                device=self.device
            )
            self.model.max_seq_length = self.max_seq_length
            self._model_loaded = True
        except Exception as e:
            logger.warning(f"Could not load {self.model_name}: {e}. Using mock encoder.")
            self._model_loaded = False

    # Physics + math concept vocabulary used for semantic mock embeddings.
    # Each term maps to a cluster index; semantically related terms share clusters.
    _CONCEPT_CLUSTERS = {
        # Cluster 0: kinematics / motion
        "velocity":0,"speed":0,"position":0,"displacement":0,"kinematics":0,
        "motion":0,"distance":0,"trajectory":0,
        # Cluster 1: dynamics / forces
        "force":1,"acceleration":1,"newton":1,"dynamics":1,"friction":1,
        "gravity":1,"mass":1,"weight":1,"normal":1,
        # Cluster 2: energy / work
        "energy":2,"work":2,"power":2,"kinetic":2,"potential":2,"joule":2,
        "conservation":2,"mechanical":2,
        # Cluster 3: momentum
        "momentum":3,"impulse":3,"collision":3,"elastic":3,"inelastic":3,
        # Cluster 4: rotation
        "torque":4,"rotation":4,"angular":4,"moment":4,"inertia":4,
        "centripetal":4,"circular":4,
        # Cluster 5: waves / oscillation
        "wave":5,"frequency":5,"amplitude":5,"period":5,"oscillation":5,
        "harmonic":5,"pendulum":5,"resonance":5,
        # Cluster 6: thermodynamics
        "temperature":6,"heat":6,"entropy":6,"thermodynamics":6,
        "pressure":6,"gas":6,"thermal":6,"boltzmann":6,
        # Cluster 7: electricity / EM
        "electric":7,"charge":7,"voltage":7,"current":7,"resistance":7,
        "magnetic":7,"field":7,"electromagnetic":7,
        # Cluster 8: calculus
        "derivative":8,"integral":8,"limit":8,"differential":8,"calculus":8,
        "function":8,"slope":8,
        # Cluster 9: quantum / modern
        "quantum":9,"photon":9,"wave-particle":9,"uncertainty":9,
        "atom":9,"electron":9,"nucleus":9,
        # Cluster 10: algebra / equations (ASSISTments math)
        "equation":10,"variable":10,"expression":10,"solve":10,"solving":10,
        "linear":10,"quadratic":10,"inequality":10,"absolute":10,"value":10,
        "simplify":10,"factor":10,"polynomial":10,"exponent":10,"radical":10,
        # Cluster 11: geometry / shapes (ASSISTments math)
        "area":11,"triangle":11,"rectangle":11,"circle":11,"parallelogram":11,
        "trapezoid":11,"perimeter":11,"circumference":11,"angle":11,
        "congruence":11,"similar":11,"pythagorean":11,"polygon":11,
        "volume":11,"surface":11,
        # Cluster 12: arithmetic / fractions (ASSISTments math)
        "fraction":12,"decimal":12,"percent":12,"ratio":12,"proportion":12,
        "addition":12,"subtraction":12,"multiplication":12,"division":12,
        "integer":12,"whole":12,"number":12,"negative":12,"positive":12,
        # Cluster 13: data / statistics (ASSISTments math)
        "mean":13,"median":13,"mode":13,"range":13,"histogram":13,
        "graph":13,"scatter":13,"probability":13,"data":13,"whisker":13,
        "box":13,"stem":13,"leaf":13,"table":13,
        # Cluster 14: number theory
        "prime":14,"factor":14,"multiple":14,"divisibility":14,"lcm":14,
        "gcf":14,"greatest":14,"least":14,"common":14,
        # Cluster 15: measurement / conversion
        "convert":15,"unit":15,"measurement":15,"scale":15,"coordinate":15,
        "midpoint":15,"line":15,"segment":15,"ray":15,"parallel":15,
        "perpendicular":15,"transversal":15,"complementary":15,"supplementary":15,
    }
    N_CLUSTERS = 16

    def _mock_encode(self, texts: List[str]) -> np.ndarray:
        """
        Semantic mock embeddings using concept-cluster keyword overlap.

        Rather than pure random vectors, each text gets an embedding that
        is the weighted sum of cluster prototype vectors.  Texts sharing
        physics vocabulary therefore produce higher cosine similarity.
        This makes NDCG meaningful even without real sentence_transformers.

        Limitation: cluster prototypes are random (fixed seed), so
        cross-cluster cosine similarities are near-zero but not exactly zero.
        """
        # Fixed random cluster prototypes (shape: n_clusters × full_dim)
        proto_rng = np.random.RandomState(0xDEADBEEF)
        prototypes = proto_rng.randn(self.N_CLUSTERS, self.full_dim).astype(np.float32)
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        prototypes /= (norms + 1e-9)

        embeddings = []
        for text in texts:
            low = text.lower()
            # Count term hits per cluster
            weights = np.zeros(self.N_CLUSTERS, dtype=np.float32)
            for term, cluster in self._CONCEPT_CLUSTERS.items():
                if term in low:
                    weights[cluster] += 1.0

            if weights.sum() < 1e-6:
                # No keyword match: small noise + stable hash jitter
                seed = sum(ord(c) for c in text[:50]) % 100000
                jitter_rng = np.random.RandomState(seed)
                emb = jitter_rng.randn(self.full_dim).astype(np.float32) * 0.05
            else:
                # Weighted blend of prototypes
                weights /= weights.sum()
                emb = (weights[:, None] * prototypes).sum(axis=0)
                # Add small per-text noise for distinctiveness
                seed = sum(ord(c) for c in text[:50]) % 100000
                noise_rng = np.random.RandomState(seed)
                emb += noise_rng.randn(self.full_dim).astype(np.float32) * 0.10

            emb /= (np.linalg.norm(emb) + 1e-9)
            embeddings.append(emb)
        return np.stack(embeddings)

    def encode(
        self,
        texts: Union[str, List[str]],
        prompt_name: Optional[str] = None,
        show_progress: bool = False
    ) -> Union[MRLEmbedding, List[MRLEmbedding]]:
        """
        Encode text(s) into MRL embeddings.

        Args:
            texts: Single string or list of strings
            prompt_name: "search_query" for queries, "search_document" for corpus chunks
            show_progress: Show tqdm progress bar
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if self._model_loaded:
            if prompt_name:
                prefixed = [f"{prompt_name}: {t}" for t in texts]
            else:
                prefixed = texts

            embeddings = self.model.encode(
                prefixed,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )
        else:
            embeddings = self._mock_encode(texts)

        results = [
            MRLEmbedding(
                full_embedding=emb.astype(np.float32),
                dims=self.dims,
                text=text
            )
            for emb, text in zip(embeddings, texts)
        ]

        return results[0] if single else results

    def encode_query(self, query: str) -> MRLEmbedding:
        """Encode a student query with search_query prefix."""
        return self.encode(query, prompt_name="search_query")

    def encode_document(self, text: str) -> MRLEmbedding:
        """Encode a document chunk with search_document prefix."""
        return self.encode(text, prompt_name="search_document")

    def encode_documents_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[MRLEmbedding]:
        """Batch encode corpus documents."""
        return self.encode(texts, prompt_name="search_document", show_progress=show_progress)

    def compute_similarity(
        self,
        emb_a: MRLEmbedding,
        emb_b: MRLEmbedding,
        dim: int
    ) -> float:
        """Cosine similarity between two embeddings at a specific dimension."""
        a = emb_a.at_dim(dim)
        b = emb_b.at_dim(dim)
        return float(np.dot(a, b))

    def compute_multiscale_similarity(
        self,
        emb_a: MRLEmbedding,
        emb_b: MRLEmbedding
    ) -> Dict[int, float]:
        """Cosine similarities at all registered dimensions."""
        return {
            dim: self.compute_similarity(emb_a, emb_b, dim)
            for dim in self.dims
        }

    def compute_dimensional_divergence(
        self,
        query_emb: MRLEmbedding,
        doc_emb: MRLEmbedding,
        coarse_dim: int = 64,
        fine_dim: int = 768
    ) -> float:
        """
        Compute dimensional divergence score between a query-document pair.

        Core signal for KARMA's metacognitive gap estimation.

        High divergence (sim_768D >> sim_64D) means the document matches at
        fine-grained vocabulary level but not at coarse conceptual level.
        This indicates potential prerequisite mismatch — the document uses
        familiar words to discuss concepts beyond the student's knowledge frontier.

        Returns:
            divergence: sim_768D - sim_64D
            Positive → vocabulary match without conceptual match (risky)
            Negative → conceptual match (safe, within domain)
        """
        sim_coarse = self.compute_similarity(query_emb, doc_emb, coarse_dim)
        sim_fine = self.compute_similarity(query_emb, doc_emb, fine_dim)
        return sim_fine - sim_coarse

    def multiscale_agreement_score(
        self,
        emb_a: MRLEmbedding,
        emb_b: MRLEmbedding
    ) -> float:
        """
        Compute cross-scale agreement between two embeddings.

        Used for prerequisite graph edge validation:
        Two chunks only form a prerequisite edge if their similarity
        is stable across 64D, 256D, and 768D — not just at 768D.

        Returns:
            agreement: float in [0,1], higher = more stable across scales
        """
        sims = self.compute_multiscale_similarity(emb_a, emb_b)
        sim_values = list(sims.values())
        mean_sim = np.mean(sim_values)
        std_sim = np.std(sim_values)
        cv = std_sim / (mean_sim + 1e-9)
        return float(1.0 / (1.0 + cv))