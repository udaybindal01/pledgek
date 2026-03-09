"""
Unified Data Loader for PLEDGE-KARMA
======================================
Auto-detects which datasets are downloaded and loads them into
a standardized format that the experiment runner can consume.

Usage:
    from data.data_loader import DataLoader

    loader = DataLoader("data/processed")
    corpus = loader.load_corpus()           # ConceptNodes + CorpusChunks
    edges = loader.load_prereq_edges()       # PrereqEdge list
    interactions = loader.load_interactions() # student_id → [Interaction]
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
    name: str
    path: Path
    available: bool
    n_items: int = 0


class DataLoader:
    """
    Unified data loader that auto-detects available datasets.

    Directory structure expected under processed_dir:
      processed_dir/
        physics_v1/           # OpenStax
          concepts.json
          chunks.json
        lecturebank/          # LectureBank
          prereq_edges.json
        assistments/          # ASSISTments
          interactions.csv
          skill_mapping.json
        junyi/                # Junyi Academy
          concepts.json
          interactions_summary.json
        mooccube/             # MOOCCube (optional)
          concepts.json
          edges.json
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.base = Path(processed_dir)
        self._available = self._detect_datasets()

    def _detect_datasets(self) -> Dict[str, DatasetInfo]:
        """Scan for available processed datasets."""
        datasets = {}

        # OpenStax (Unified full corpus)
        for book in ["openstax_full", "physics_v1"]:
            p = self.base / book
            avail = (p / "concepts.json").exists() and (p / "chunks.json").exists()
            datasets[book] = DatasetInfo(book, p, avail)

        # LectureBank
        p = self.base / "lecturebank"
        avail = (p / "prereq_edges.json").exists()
        datasets["lecturebank"] = DatasetInfo("lecturebank", p, avail)

        # ASSISTments
        p = self.base / "assistments"
        avail = (p / "interactions.csv").exists()
        datasets["assistments"] = DatasetInfo("assistments", p, avail)

        # Junyi
        p = self.base / "junyi"
        avail = (p / "concepts.json").exists()
        datasets["junyi"] = DatasetInfo("junyi", p, avail)

        # MOOCCube
        p = self.base / "mooccube"
        avail = (p / "concepts.json").exists()
        datasets["mooccube"] = DatasetInfo("mooccube", p, avail)

        available = [k for k, v in datasets.items() if v.available]
        logger.info(f"Available datasets: {available}")
        return datasets

    def get_available_datasets(self) -> List[str]:
        """Return names of available datasets."""
        return [k for k, v in self._available.items() if v.available]

    # ── Corpus loading ──────────────────────────────────────────────────

    def load_corpus(
        self,
        sources: Optional[List[str]] = None,
    ) -> Tuple[List[ConceptNode], List[CorpusChunk]]:
        """
        Load textbook corpus (concepts + chunks).

        Args:
            sources: List of book keys, e.g. ["physics_v1"].
                     If None, loads all available OpenStax books.
        """
        if sources is None:
            # Prefer the full corpus if available, otherwise fallback to whatever is there
            if self._available.get("openstax_full", DatasetInfo("", Path(), False)).available:
                sources = ["openstax_full"]
            else:
                sources = [k for k in ["physics_v1", "physics_v2"]
                           if self._available.get(k, DatasetInfo("", Path(), False)).available]

        if not sources:
            logger.warning("No corpus data found. Use data/prepare_data.py to download.")
            return [], []

        all_concepts = []
        all_chunks = []

        for source in sources:
            info = self._available.get(source)
            if not info or not info.available:
                logger.warning(f"Dataset '{source}' not found, skipping")
                continue

            concepts = self._load_concepts_json(info.path / "concepts.json")
            chunks = self._load_chunks_json(info.path / "chunks.json")
            all_concepts.extend(concepts)
            all_chunks.extend(chunks)
            logger.info(f"Loaded {source}: {len(concepts)} concepts, {len(chunks)} chunks")

        return all_concepts, all_chunks

    def _load_concepts_json(self, path: Path) -> List[ConceptNode]:
        """Load concepts from JSON."""
        with open(path) as f:
            raw = json.load(f)
        concepts = []
        for d in raw:
            concepts.append(ConceptNode(
                concept_id=d["concept_id"],
                name=d["name"],
                description=d.get("description", d["name"]),
                source_chunk_ids=d.get("source_chunk_ids", []),
                depth_level=d.get("depth_level", 1),
                chapter_order=d.get("chapter_order", 0),
                subject=d.get("subject", "general"),
                tags=d.get("tags", []),
            ))
        return concepts

    def _load_chunks_json(self, path: Path) -> List[CorpusChunk]:
        """Load chunks from JSON."""
        with open(path) as f:
            raw = json.load(f)
        chunks = []
        for d in raw:
            chunks.append(CorpusChunk(
                chunk_id=d["chunk_id"],
                text=d["text"],
                concept_ids=d.get("concept_ids", []),
                prerequisite_concept_ids=d.get("prerequisite_concept_ids", []),
                depth_level=d.get("depth_level", 1),
                chapter_order=d.get("chapter_order", 0),
                subject=d.get("subject", "general"),
                source=d.get("source", "unknown"),
                metadata=d.get("metadata", {}),
            ))
        return chunks

    # ── Prerequisite edges ──────────────────────────────────────────────

    def load_prereq_edges(
        self,
        sources: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Load prerequisite edges as (source_id, target_id, confidence) tuples.

        Sources: "lecturebank", "mooccube"
        """
        if sources is None:
            sources = ["lecturebank", "mooccube"]

        all_edges = []

        for source in sources:
            info = self._available.get(source)
            if not info or not info.available:
                continue

            edge_file = info.path / "prereq_edges.json"
            if not edge_file.exists():
                continue

            with open(edge_file) as f:
                raw = json.load(f)

            for edge in raw:
                all_edges.append((
                    edge["source_id"],
                    edge["target_id"],
                    edge.get("confidence", 0.8),
                ))

            logger.info(f"Loaded {len(raw)} prereq edges from {source}")

        return all_edges

    # ── Student interactions ────────────────────────────────────────────

    def load_interactions(
        self,
        sources: Optional[List[str]] = None,
        max_students: int = 1000,
        max_rows_per_source: int = 500000,
    ) -> Dict[str, List[Interaction]]:
        """
        Load student interaction logs from available sources.

        Returns: Dict[student_id → List[Interaction]], sorted by timestamp.
        """
        if sources is None:
            sources = ["assistments", "junyi"]

        all_interactions: Dict[str, List[Interaction]] = {}

        for source in sources:
            info = self._available.get(source)
            if not info or not info.available:
                continue

            if source == "assistments":
                ints = self._load_assistments_interactions(
                    info.path, max_students, max_rows_per_source
                )
            elif source == "junyi":
                ints = self._load_junyi_interactions(
                    info.path, max_students, max_rows_per_source
                )
            else:
                continue

            # Merge into all_interactions with prefixed student IDs
            for uid, int_list in ints.items():
                prefixed_uid = f"{source}_{uid}"
                all_interactions[prefixed_uid] = int_list

            logger.info(
                f"Loaded {len(ints)} students from {source} "
                f"({sum(len(v) for v in ints.values())} interactions)"
            )

        return all_interactions

    def _load_assistments_interactions(
        self, path: Path, max_students: int, max_rows: int
    ) -> Dict[str, List[Interaction]]:
        """Load ASSISTments interactions from processed CSV."""
        csv_path = path / "interactions.csv"
        if not csv_path.exists():
            return {}

        # Load skill mapping if available
        skill_map_path = path / "skill_mapping.json"
        skill_map = {}
        if skill_map_path.exists():
            with open(skill_map_path) as f:
                skill_map = json.load(f)

        interactions: Dict[str, List[Interaction]] = {}
        row_count = 0

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                if row_count > max_rows:
                    break

                uid = row.get("user_id", row.get("student_id", "")).strip()
                if not uid:
                    continue
                if len(interactions) >= max_students and uid not in interactions:
                    continue

                # Parse fields
                skill = row.get("skill_name", row.get("skill_id", "")).strip()
                correct_raw = row.get("correct", "0").strip()
                correct = correct_raw in ("1", "true", "True")
                hint_count = int(row.get("hint_count", "0") or "0")
                attempt_count = int(row.get("attempt_count", "1") or "1")

                # Parse timestamp
                ts_raw = row.get("start_time", row.get("timestamp", "")).strip()
                try:
                    timestamp = datetime.fromisoformat(ts_raw) if ts_raw else \
                        datetime(2012, 9, 1) + timedelta(seconds=row_count)
                except (ValueError, TypeError):
                    timestamp = datetime(2012, 9, 1) + timedelta(seconds=row_count)

                # Map skill to concept ID
                concept_id = skill_map.get(skill, f"assist_{skill}")

                # Quality from correctness + hints
                if correct and hint_count == 0:
                    quality = 0.95
                elif correct:
                    quality = 0.6
                else:
                    quality = 0.2

                interaction = Interaction(
                    interaction_id=f"assist_{uid}_{row_count}",
                    timestamp=timestamp,
                    query=f"exercise: {skill}",
                    concept_ids=[concept_id],
                    correct=correct,
                    response_quality=quality,
                    mrl_divergence=0.0,
                )
                interactions.setdefault(uid, []).append(interaction)

        for uid in interactions:
            interactions[uid].sort(key=lambda x: x.timestamp)

        return interactions

    def _load_junyi_interactions(
        self, path: Path, max_students: int, max_rows: int
    ) -> Dict[str, List[Interaction]]:
        """Load Junyi interactions (delegates to JunyiProcessor if raw data, else loads processed)."""
        # Try loading from raw data via processor
        raw_dir = Path("data/raw/junyi")
        if (raw_dir / "Log_Problem.csv").exists():
            from models.junyi_processor import JunyiProcessor
            proc = JunyiProcessor(str(raw_dir), max_students=max_students)
            proc.load_content_metadata()
            return proc.load_interactions(max_rows=max_rows)

        # Otherwise return empty
        logger.info("Junyi raw data not found, skipping")
        return {}

    # ── Summary ─────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return summary of all available and loaded data."""
        return {
            name: {
                "available": info.available,
                "path": str(info.path),
            }
            for name, info in self._available.items()
        }
