"""
Junyi Academy Dataset Processor
=================================
Processes Junyi Academy CSV files into KARMA-compatible interaction sequences.

Expected files (from Kaggle: junyi-academy-online-learning-activity-dataset):
  Log_Problem.csv    — problem attempt logs (16M+ rows)
  Info_Content.csv   — exercise/content metadata

Key columns in Log_Problem.csv:
  uuid, ucid, problem_id, exercise_id, topic_id,
  timestamp_TW, correct, hint_count, attempt_count,
  time_taken, review_flag

Key columns in Info_Content.csv:
  ucid, content_pretty_name, topic_id, level

This processor:
  1. Loads exercise metadata → ConceptNode objects
  2. Loads student interaction logs → Interaction objects for KARMA
  3. Preserves timestamps for forgetting curve validation
"""

import csv
import logging
import hashlib
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph.graph_builder import ConceptNode, CorpusChunk
from karma.estimator import Interaction

logger = logging.getLogger(__name__)


def _generate_id(text: str, prefix: str = "") -> str:
    """Generate a deterministic short ID."""
    h = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{prefix}_{h}" if prefix else h


@dataclass
class JunyiExercise:
    """Metadata for a single Junyi Academy exercise."""
    exercise_id: str
    name: str
    topic_id: str
    topic_name: str
    level: int            # Difficulty level (0-based)
    prerequisites: List[str] = field(default_factory=list)


class JunyiProcessor:
    """
    Processes Junyi Academy data for PLEDGE-KARMA.

    Outputs:
      - ConceptNodes from exercise topics (for knowledge graph)
      - Student interaction sequences (for KARMA training/validation)
      - Temporal interaction data (for forgetting curve validation)
    """

    def __init__(self, data_dir: str, max_students: int = 5000):
        self.data_dir = Path(data_dir)
        self.max_students = max_students
        self.exercises: Dict[str, JunyiExercise] = {}
        self.topic_to_concept: Dict[str, str] = {}

    def load_content_metadata(self) -> List[ConceptNode]:
        """Load Info_Content.csv and create concept nodes from topics."""
        path = self.data_dir / "Info_Content.csv"
        if not path.exists():
            logger.warning(f"Junyi Info_Content.csv not found at {path}")
            return []

        topics: Dict[str, Dict] = {}

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ucid = row.get("ucid", "").strip()
                name = row.get("content_pretty_name", "").strip()
                topic_id = row.get("topic_id", "").strip()
                level = int(row.get("level", "0") or "0")

                if not ucid or not name:
                    continue

                self.exercises[ucid] = JunyiExercise(
                    exercise_id=ucid,
                    name=name,
                    topic_id=topic_id,
                    topic_name=name,
                    level=level,
                )

                # Group by topic
                if topic_id and topic_id not in topics:
                    topics[topic_id] = {
                        "names": [],
                        "level": level,
                        "exercises": []
                    }
                if topic_id:
                    topics[topic_id]["names"].append(name)
                    topics[topic_id]["exercises"].append(ucid)

        # Create concept nodes from topics
        concepts = []
        for i, (topic_id, info) in enumerate(sorted(topics.items())):
            # Use the most common exercise name as the concept name
            concept_name = info["names"][0] if info["names"] else topic_id
            concept_id = _generate_id(topic_id, "junyi")
            self.topic_to_concept[topic_id] = concept_id

            # Map difficulty levels to depth
            depth = min(info["level"], 2)

            concepts.append(ConceptNode(
                concept_id=concept_id,
                name=concept_name,
                description=f"Junyi Academy topic: {concept_name}",
                source_chunk_ids=info["exercises"],
                depth_level=depth,
                chapter_order=i * 10,
                subject="math",
                tags=["junyi", f"level_{info['level']}"],
            ))

        logger.info(f"Junyi: {len(concepts)} concepts from {len(self.exercises)} exercises")
        return concepts

    def load_interactions(
        self,
        max_rows: Optional[int] = None,
    ) -> Dict[str, List[Interaction]]:
        """
        Load Log_Problem.csv and convert to KARMA Interaction objects.

        Returns: Dict[student_id → List[Interaction]] sorted by timestamp.
        """
        path = self.data_dir / "Log_Problem.csv"
        if not path.exists():
            logger.warning(f"Junyi Log_Problem.csv not found at {path}")
            return {}

        student_interactions: Dict[str, List[Interaction]] = {}
        row_count = 0
        skipped = 0

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                if max_rows and row_count > max_rows:
                    break

                uuid = row.get("uuid", "").strip()
                if not uuid:
                    skipped += 1
                    continue

                # Limit number of students
                if (len(student_interactions) >= self.max_students
                        and uuid not in student_interactions):
                    continue

                # Parse fields
                exercise_id = row.get("ucid", row.get("exercise_id", "")).strip()
                topic_id = row.get("topic_id", "").strip()
                correct_raw = row.get("correct", "").strip().lower()
                correct = correct_raw in ("true", "1", "yes")
                hint_count = int(row.get("hint_count", "0") or "0")
                attempt_count = int(row.get("attempt_count", "1") or "1")
                time_taken = float(row.get("time_taken", "0") or "0")

                # Parse timestamp
                ts_str = row.get("timestamp_TW", "").strip()
                try:
                    if ts_str:
                        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    else:
                        timestamp = datetime(2018, 8, 1) + timedelta(seconds=row_count)
                except (ValueError, TypeError):
                    timestamp = datetime(2018, 8, 1) + timedelta(seconds=row_count)

                # Map to concept IDs
                concept_id = self.topic_to_concept.get(
                    topic_id,
                    _generate_id(exercise_id or topic_id, "junyi")
                )

                # Compute response quality from hints and attempts
                # Fewer hints + fewer attempts + correct = higher quality
                if correct and hint_count == 0 and attempt_count <= 1:
                    response_quality = 0.95
                elif correct and hint_count == 0:
                    response_quality = 0.7
                elif correct:
                    response_quality = 0.5
                else:
                    response_quality = 0.2

                interaction = Interaction(
                    interaction_id=f"junyi_{uuid}_{row_count}",
                    timestamp=timestamp,
                    query=f"exercise: {exercise_id}",
                    concept_ids=[concept_id],
                    correct=correct,
                    response_quality=response_quality,
                    mrl_divergence=0.0,  # No MRL data from Junyi
                )

                student_interactions.setdefault(uuid, []).append(interaction)

        # Sort each student's interactions by timestamp
        for uid in student_interactions:
            student_interactions[uid].sort(key=lambda x: x.timestamp)

        logger.info(
            f"Junyi: {len(student_interactions)} students, "
            f"{sum(len(v) for v in student_interactions.values())} interactions "
            f"({skipped} skipped, {row_count} rows read)"
        )
        return student_interactions

    def compute_temporal_stats(
        self,
        interactions: Dict[str, List[Interaction]]
    ) -> Dict:
        """
        Compute temporal statistics for forgetting curve validation.

        Returns stats on inter-review intervals, retention patterns, etc.
        """
        all_gaps = []
        correct_after_gap = {1: [], 3: [], 7: [], 14: [], 30: []}

        for uid, ints in interactions.items():
            concept_last_seen: Dict[str, Tuple[datetime, bool]] = {}

            for interaction in ints:
                for concept_id in interaction.concept_ids:
                    if concept_id in concept_last_seen:
                        last_time, last_correct = concept_last_seen[concept_id]
                        gap_days = (
                            interaction.timestamp - last_time
                        ).total_seconds() / 86400.0

                        if gap_days > 0:
                            all_gaps.append(gap_days)
                            # Bin by gap length
                            for threshold in sorted(correct_after_gap.keys()):
                                if gap_days <= threshold:
                                    correct_after_gap[threshold].append(
                                        1 if interaction.correct else 0
                                    )
                                    break

                    concept_last_seen[concept_id] = (
                        interaction.timestamp,
                        interaction.correct
                    )

        stats = {
            "n_students": len(interactions),
            "n_total_interactions": sum(len(v) for v in interactions.values()),
            "median_gap_days": float(np.median(all_gaps)) if all_gaps else 0,
            "mean_gap_days": float(np.mean(all_gaps)) if all_gaps else 0,
        }

        for threshold, outcomes in correct_after_gap.items():
            if outcomes:
                stats[f"retention_within_{threshold}d"] = float(np.mean(outcomes))
            else:
                stats[f"retention_within_{threshold}d"] = None

        return stats

    def save_processed(self, output_dir: str) -> None:
        """Save processed data to JSON files."""
        out = Path(output_dir) / "junyi"
        out.mkdir(parents=True, exist_ok=True)

        concepts = self.load_content_metadata()
        interactions = self.load_interactions()

        # Save concepts
        with open(out / "concepts.json", "w") as f:
            json.dump([{
                "concept_id": c.concept_id,
                "name": c.name,
                "description": c.description,
                "depth_level": c.depth_level,
                "chapter_order": c.chapter_order,
                "subject": c.subject,
                "tags": c.tags,
            } for c in concepts], f, indent=2)

        # Save interaction summary (full logs are too large for JSON)
        summary = {
            "n_students": len(interactions),
            "n_interactions": sum(len(v) for v in interactions.values()),
            "students": {
                uid: len(ints)
                for uid, ints in list(interactions.items())[:100]
            }
        }
        with open(out / "interactions_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save temporal stats
        stats = self.compute_temporal_stats(interactions)
        with open(out / "temporal_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Junyi data saved to {out}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/junyi")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--max-students", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    proc = JunyiProcessor(args.data_dir, max_students=args.max_students)
    proc.save_processed(args.output)
