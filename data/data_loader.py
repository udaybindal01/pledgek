"""
Unified Data Loader for PLEDGE-KARMA
======================================
Auto-detects which datasets are downloaded and loads them into
a standardized format that the experiment runner can consume.

Three-Axis Evaluation Strategy:
  Axis 1 — Prereq Graph Quality:
      loader.load_corpus()          → OpenStax multi-subject corpus
      loader.load_prereq_edges()    → LectureBank + OpenStax ordering edges
  Axis 2 — KT / Forgetting:
      loader.load_interactions()    → EdNet (primary) or ASSISTments (fallback)
  Axis 3 — End-to-End Retrieval:
      loader.load_mooccube()        → MOOCCube graph + student logs

Usage:
    from data.data_loader import DataLoader

    loader = DataLoader("data/processed")

    # Axis 1: prereq graph evaluation
    corpus = loader.load_corpus()
    edges  = loader.load_prereq_edges(sources=["lecturebank"])

    # Axis 2: KT evaluation (EdNet primary, ASSISTments fallback)
    interactions = loader.load_interactions(prefer_ednet=True)

    # Axis 3: end-to-end
    mc_concepts, mc_edges, mc_logs = loader.load_mooccube()
"""

import json
import csv
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph.graph_builder import ConceptNode, CorpusChunk
from karma.estimator import Interaction

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Info about an available dataset."""
    name:      str
    path:      Path
    available: bool
    n_items:   int = 0


class DataLoader:
    """
    Unified data loader that auto-detects available datasets.

    Directory structure under processed_dir:
      processed_dir/
        openstax_full/          ← multi-subject corpus (Axis 1)
          concepts.json
          chunks.json
        lecturebank/            ← prereq ground truth (Axis 1)
          prereq_edges.json
          concepts.json
          negative_pairs.json
        ednet/                  ← KT evaluation, PRIMARY (Axis 2)
          interactions.json
          student_index.json
          questions.json
          skill_mapping.json
          corpus_chunks.json
          stats.json
        assistments/            ← KT evaluation, FALLBACK (Axis 2)
          interactions.csv
          skill_mapping.json
        mooccube/               ← end-to-end evaluation (Axis 3)
          concepts.json
          edges.json
          student_logs.json
        junyi/                  ← supplementary
          concepts.json
          interactions_summary.json
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.base      = Path(processed_dir)
        self._available = self._detect_datasets()

    def _detect_datasets(self) -> Dict[str, DatasetInfo]:
        """Scan for available processed datasets."""
        datasets = {}

        # OpenStax (multi-subject preferred, physics fallback)
        for book in ["openstax_full", "physics_v1", "physics_v2"]:
            p     = self.base / book
            avail = (p / "concepts.json").exists() and (p / "chunks.json").exists()
            datasets[book] = DatasetInfo(book, p, avail)

        # LectureBank (Axis 1: prereq graph ground truth)
        p     = self.base / "lecturebank"
        avail = (p / "prereq_edges.json").exists()
        datasets["lecturebank"] = DatasetInfo("lecturebank", p, avail)

        # EdNet (Axis 2: KT/forgetting — PRIMARY)
        p     = self.base / "ednet"
        avail = (p / "interactions.json").exists() and (p / "student_index.json").exists()
        datasets["ednet"] = DatasetInfo("ednet", p, avail)

        # ASSISTments (Axis 2: KT — FALLBACK)
        p     = self.base / "assistments"
        avail = (p / "interactions.csv").exists()
        datasets["assistments"] = DatasetInfo("assistments", p, avail)

        # Junyi (supplementary)
        p     = self.base / "junyi"
        avail = (p / "concepts.json").exists()
        datasets["junyi"] = DatasetInfo("junyi", p, avail)

        # MOOCCube (Axis 3: end-to-end evaluation)
        p     = self.base / "mooccube"
        avail = (p / "concepts.json").exists()
        datasets["mooccube"] = DatasetInfo("mooccube", p, avail)

        available = [k for k, v in datasets.items() if v.available]
        logger.info(f"Available datasets: {available}")
        return datasets

    def get_available_datasets(self) -> List[str]:
        """Return names of available datasets."""
        return [k for k, v in self._available.items() if v.available]

    def get_axis_availability(self) -> Dict[str, bool]:
        """
        Check which evaluation axes have data available.
        Used by the experiment runner to decide which axes to run.
        """
        has_corpus   = (
            self._available.get("openstax_full",  DatasetInfo("", Path(), False)).available or
            self._available.get("physics_v1",     DatasetInfo("", Path(), False)).available
        )
        has_lb       = self._available.get("lecturebank",  DatasetInfo("", Path(), False)).available
        has_ednet    = self._available.get("ednet",        DatasetInfo("", Path(), False)).available
        has_assist   = self._available.get("assistments",  DatasetInfo("", Path(), False)).available
        has_mooccube = self._available.get("mooccube",     DatasetInfo("", Path(), False)).available

        return {
            "axis_1_prereq":      has_corpus and has_lb,
            "axis_2_kt":          has_ednet or has_assist,
            "axis_2_kt_ednet":    has_ednet,
            "axis_2_kt_assist":   has_assist,
            "axis_3_endtoend":    has_mooccube,
        }

    # ── Axis 1: Corpus loading ──────────────────────────────────────────────

    def load_corpus(
        self,
        sources: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
    ) -> Tuple[List[ConceptNode], List[CorpusChunk]]:
        """
        Load textbook corpus (concepts + chunks).

        Args:
            sources:  Dataset keys to load. If None, auto-detects best available.
                      Prefer "openstax_full" (multi-subject) over "physics_v1".
            subjects: Filter by subject (e.g., ["physics", "calculus"]).
                      None = load all subjects.
        Returns:
            (concepts, chunks) — merged from all specified sources.
        """
        if sources is None:
            if self._available.get(
                "openstax_full", DatasetInfo("", Path(), False)
            ).available:
                sources = ["openstax_full"]
            else:
                sources = [
                    k for k in ["physics_v1", "physics_v2"]
                    if self._available.get(k, DatasetInfo("", Path(), False)).available
                ]

        if not sources:
            logger.warning("No corpus data found. Run data/prepare_data.py first.")
            return [], []

        all_concepts: List[ConceptNode] = []
        all_chunks:   List[CorpusChunk] = []

        for source in sources:
            info = self._available.get(source)
            if not info or not info.available:
                logger.warning(f"Dataset '{source}' not found, skipping")
                continue

            concepts = self._load_concepts_json(info.path / "concepts.json")
            chunks   = self._load_chunks_json(info.path / "chunks.json")

            # Subject filter
            if subjects:
                concepts = [c for c in concepts if c.subject in subjects]
                valid_cids = {c.concept_id for c in concepts}
                chunks = [
                    ch for ch in chunks
                    if ch.subject in subjects or
                    any(cid in valid_cids for cid in ch.concept_ids)
                ]

            all_concepts.extend(concepts)
            all_chunks.extend(chunks)
            logger.info(
                f"Loaded {source}: {len(concepts)} concepts, "
                f"{len(chunks)} chunks"
                + (f" (subjects={subjects})" if subjects else "")
            )

        # Log subject distribution
        from collections import Counter
        subj_dist = Counter(c.subject for c in all_concepts)
        logger.info(f"Corpus subject distribution: {dict(subj_dist)}")

        return all_concepts, all_chunks

    def _load_concepts_json(self, path: Path) -> List[ConceptNode]:
        with open(path) as f:
            raw = json.load(f)
        return [
            ConceptNode(
                concept_id=d["concept_id"],
                name=d["name"],
                description=d.get("description", d["name"]),
                source_chunk_ids=d.get("source_chunk_ids", []),
                depth_level=d.get("depth_level", 1),
                chapter_order=d.get("chapter_order", 0),
                subject=d.get("subject", "general"),
                tags=d.get("tags", []),
            )
            for d in raw
        ]

    def _load_chunks_json(self, path: Path) -> List[CorpusChunk]:
        with open(path) as f:
            raw = json.load(f)
        return [
            CorpusChunk(
                chunk_id=d["chunk_id"],
                text=d["text"],
                concept_ids=d.get("concept_ids", []),
                prerequisite_concept_ids=d.get("prerequisite_concept_ids", []),
                depth_level=d.get("depth_level", 1),
                chapter_order=d.get("chapter_order", 0),
                subject=d.get("subject", "general"),
                source=d.get("source", "unknown"),
                metadata=d.get("metadata", {}),
            )
            for d in raw
        ]

    # ── Axis 1: Prerequisite edges ──────────────────────────────────────────

    def load_prereq_edges(
        self,
        sources: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Load prerequisite edges as (source_id, target_id, confidence) tuples.

        For Axis 1 evaluation:
          - LectureBank: human-verified CS/NLP edges (ground truth)
          - MOOCCube:    semi-automatic edges with confidence scores

        Args:
            sources: Which sources to include. Defaults to all available.
        Returns:
            Merged list of (src_id, tgt_id, confidence) edges.
        """
        if sources is None:
            sources = ["lecturebank", "mooccube"]

        all_edges: List[Tuple[str, str, float]] = []

        for source in sources:
            info = self._available.get(source)
            if not info or not info.available:
                continue

            edges_path = info.path / "prereq_edges.json"
            if not edges_path.exists():
                edges_path = info.path / "edges.json"
            if not edges_path.exists():
                continue

            with open(edges_path) as f:
                raw = json.load(f)

            for e in raw:
                src = e.get("source_id", "")
                tgt = e.get("target_id", "")
                conf = float(e.get("confidence", 0.7))
                if src and tgt:
                    all_edges.append((src, tgt, conf))

            logger.info(f"Loaded {len(raw)} prereq edges from {source}")

        return all_edges

    def load_lecturebank_for_eval(self) -> Dict:
        """
        Load LectureBank in evaluation format.
        Returns both positive edges and negative pairs for Axis 1 F1 evaluation.
        """
        lb = self._available.get("lecturebank")
        if not lb or not lb.available:
            return {"positive_edges": [], "negative_pairs": [], "concepts": []}

        result = {"concepts": [], "positive_edges": [], "negative_pairs": []}

        concepts_path = lb.path / "concepts.json"
        if concepts_path.exists():
            with open(concepts_path) as f:
                result["concepts"] = json.load(f)

        edges_path = lb.path / "prereq_edges.json"
        if edges_path.exists():
            with open(edges_path) as f:
                result["positive_edges"] = json.load(f)

        neg_path = lb.path / "negative_pairs.json"
        if neg_path.exists():
            with open(neg_path) as f:
                result["negative_pairs"] = json.load(f)

        return result

    # ── Axis 2: Student interactions ───────────────────────────────────────

    def load_interactions(
        self,
        prefer_ednet:   bool = True,
        max_students:   Optional[int] = None,
        source_override: Optional[str] = None,
    ) -> Tuple[str, Dict[str, List[Dict]]]:
        """
        Load student interactions for KT/forgetting evaluation (Axis 2).

        Automatically selects EdNet (primary) or ASSISTments (fallback).

        Args:
            prefer_ednet:    Use EdNet when available (recommended).
            max_students:    Cap number of students.
            source_override: Force a specific source ("ednet" or "assistments").
        Returns:
            (source_name, {student_id: [interaction_dicts]})
        """
        avail = self.get_axis_availability()

        if source_override:
            use_ednet = (source_override == "ednet")
        elif prefer_ednet and avail["axis_2_kt_ednet"]:
            use_ednet = True
        else:
            use_ednet = False

        if use_ednet:
            logger.info("Loading EdNet interactions (Axis 2 PRIMARY)...")
            interactions = self._load_ednet_interactions(max_students)
            return "ednet", interactions
        elif avail["axis_2_kt_assist"]:
            logger.info("Loading ASSISTments interactions (Axis 2 FALLBACK)...")
            interactions = self._load_assistments_interactions(max_students)
            return "assistments", interactions
        else:
            logger.warning("No interaction data available. Run data/prepare_data.py.")
            return "none", {}

    def _load_ednet_interactions(
        self, max_students: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Load processed EdNet interactions.
        Each interaction has: concept_id, correct, timestamp, mrl_divergence,
                              response_quality, question_text, elapsed_ms.
        """
        ednet_info = self._available["ednet"]

        with open(ednet_info.path / "interactions.json") as f:
            all_interactions = json.load(f)

        with open(ednet_info.path / "student_index.json") as f:
            student_index = json.load(f)

        if max_students:
            student_ids   = list(student_index.keys())[:max_students]
            student_index = {sid: student_index[sid] for sid in student_ids}

        result: Dict[str, List[Dict]] = {}
        for uid, idxs in student_index.items():
            result[uid] = [all_interactions[i] for i in idxs]

        mrl_count = sum(
            1 for ilist in result.values()
            for i in ilist if i.get("mrl_divergence", 0.0) != 0.0
        )
        total = sum(len(v) for v in result.values())
        logger.info(
            f"EdNet: {len(result)} students, {total} interactions, "
            f"MRL coverage={mrl_count/max(total,1):.1%}"
        )
        return result

    def _load_assistments_interactions(
        self, max_students: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Load ASSISTments interactions.
        Note: mrl_divergence may be 0.0 unless inject_mrl was used in prepare_data.
        """
        assist_info = self._available["assistments"]
        result: Dict[str, List[Dict]] = {}

        with open(assist_info.path / "interactions.csv",
                  encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("user_id", "").strip()
                if not uid:
                    continue
                if max_students and uid not in result and len(result) >= max_students:
                    continue

                skill   = row.get("skill_name", "").strip()
                correct = int(row.get("correct", "0").strip() == "1")
                mrl_div = float(row.get("mrl_divergence", "0.0") or "0.0")

                if uid not in result:
                    result[uid] = []
                result[uid].append({
                    "concept_id":       skill,
                    "skill_name":       skill,
                    "correct":          correct,
                    "timestamp":        row.get("timestamp", 0),
                    "mrl_divergence":   mrl_div,
                    "response_quality": float(correct),
                    "hint_count":       int(row.get("hint_count", 0) or 0),
                    "attempt_count":    int(row.get("attempt_count", 1) or 1),
                })

        mrl_count = sum(
            1 for ilist in result.values()
            for i in ilist if i.get("mrl_divergence", 0.0) != 0.0
        )
        total = sum(len(v) for v in result.values())
        logger.info(
            f"ASSISTments: {len(result)} students, {total} interactions, "
            f"MRL coverage={mrl_count/max(total,1):.1%}"
            + (" (WARNING: mrl_divergence=0 everywhere — run with --inject-mrl)"
               if mrl_count == 0 else "")
        )
        return result

    # ── Axis 3: MOOCCube end-to-end ────────────────────────────────────────

    def load_mooccube(
        self,
    ) -> Tuple[List[Dict], List[Tuple[str, str, float]], Dict[str, List[Dict]]]:
        """
        Load MOOCCube for end-to-end Axis 3 evaluation.

        Returns:
            (concepts, prereq_edges, student_logs)
            where student_logs = {student_id: [interaction_dicts]}
        """
        mc_info = self._available.get("mooccube")
        if not mc_info or not mc_info.available:
            logger.warning("MOOCCube not available. Run data/prepare_data.py --only mooccube")
            return [], [], {}

        with open(mc_info.path / "concepts.json") as f:
            concepts = json.load(f)

        edges: List[Tuple[str, str, float]] = []
        edges_path = mc_info.path / "edges.json"
        if edges_path.exists():
            with open(edges_path) as f:
                raw_edges = json.load(f)
            for e in raw_edges:
                edges.append((
                    e.get("source_id", ""),
                    e.get("target_id", ""),
                    float(e.get("confidence", 0.7)),
                ))

        student_logs: Dict[str, List[Dict]] = {}
        logs_path = mc_info.path / "student_logs.json"
        if logs_path.exists():
            with open(logs_path) as f:
                raw_logs = json.load(f)
            for interaction in raw_logs:
                uid = interaction.get("user_id", "")
                if uid not in student_logs:
                    student_logs[uid] = []
                student_logs[uid].append({
                    "concept_id":     interaction.get("concept_id", ""),
                    "correct":        int(interaction.get("correct", 0)),
                    "watch_ratio":    float(interaction.get("watch_ratio", 0.5)),
                    "timestamp":      int(interaction.get("timestamp", 0)),
                    "mrl_divergence": float(interaction.get("mrl_divergence", 0.0)),
                    "response_quality": float(interaction.get("watch_ratio", 0.5)),
                })

        logger.info(
            f"MOOCCube: {len(concepts)} concepts, {len(edges)} prereq edges, "
            f"{len(student_logs)} students"
        )
        return concepts, edges, student_logs

    # ── Legacy compatibility: load_interactions() old signature ─────────────

    def load_interactions_legacy(
        self,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, List[Interaction]]:
        """
        Load student interactions as KARMA Interaction objects (legacy format).
        Prefers EdNet, falls back to ASSISTments.
        """
        source_name, student_logs = self.load_interactions(prefer_ednet=True)
        if not student_logs:
            return {}

        result: Dict[str, List[Interaction]] = {}
        base = datetime(2024, 1, 1)

        for uid, interactions in student_logs.items():
            karma_interactions = []
            for i, item in enumerate(
                sorted(interactions, key=lambda x: x.get("timestamp", i))
            ):
                raw_ts = item.get("timestamp", 0)
                try:
                    ts = datetime.fromtimestamp(float(raw_ts)) if raw_ts else (
                        base + timedelta(hours=i)
                    )
                except (ValueError, OSError):
                    ts = base + timedelta(hours=i)

                karma_interactions.append(Interaction(
                    interaction_id=f"{uid}_{i}",
                    timestamp=ts,
                    query=item.get("question_text", f"[{item.get('concept_id', '')}]"),
                    concept_ids=[item.get("concept_id", "unknown")],
                    correct=bool(item.get("correct", 0)),
                    response_quality=float(item.get("response_quality", 0.5)),
                    mrl_divergence=float(item.get("mrl_divergence", 0.0)),
                ))
            result[uid] = karma_interactions

        return result