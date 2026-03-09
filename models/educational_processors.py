"""
Data Processors for PLEDGE-KARMA

Handles ingestion of:
  - OpenStax textbooks (primary corpus)
  - CK-12 Flexbooks (depth-level validation)
  - ASSISTments / XES3G5M (student interaction logs)
  - MOOCCube (prerequisite annotations + interactions)
  - LectureBank (human-verified prerequisite edges)

Each processor outputs standardized ConceptNode and CorpusChunk objects
compatible with the KnowledgeGraphBuilder.
"""

import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from datetime import datetime

from knowledge_graph.graph_builder import ConceptNode, CorpusChunk

# Lazy import to avoid circular dependency — Interaction is only needed in
# AssistmentsProcessor.to_karma_interactions(), loaded on first use.
def _get_interaction_class():
    from karma.estimator import Interaction
    return Interaction

logger = logging.getLogger(__name__)


def generate_id(text: str, prefix: str = "") -> str:
    """Generate a stable short ID from text content."""
    hash_val = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_val}" if prefix else hash_val


class TextChunker:
    """
    Semantic-aware text chunker.

    Splits educational text into retrievable chunks that:
    1. Respect sentence boundaries
    2. Target a specific token length
    3. Maintain context overlap between chunks
    4. Preserve paragraph/section structure
    """

    def __init__(
        self,
        chunk_size: int = 300,     # Target words per chunk
        overlap: int = 50,          # Overlap words between chunks
        min_chunk_size: int = 50
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        chunks = []
        current_chunk_sents = []
        current_word_count = 0

        for sent in sentences:
            word_count = len(sent.split())

            if current_word_count + word_count > self.chunk_size and current_chunk_sents:
                chunk_text = " ".join(current_chunk_sents)
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": metadata or {}
                    })

                # Keep overlap
                overlap_sents = []
                overlap_words = 0
                for s in reversed(current_chunk_sents):
                    if overlap_words + len(s.split()) <= self.overlap:
                        overlap_sents.insert(0, s)
                        overlap_words += len(s.split())
                    else:
                        break
                current_chunk_sents = overlap_sents
                current_word_count = overlap_words

            current_chunk_sents.append(sent)
            current_word_count += word_count

        # Last chunk
        if current_chunk_sents:
            chunk_text = " ".join(current_chunk_sents)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata or {}
                })

        return chunks


class OpenStaxProcessor:
    """
    Processes OpenStax textbooks into ConceptNodes and CorpusChunks.

    OpenStax JSON structure (from their API or scraped HTML):
    {
        "title": "University Physics Volume 1",
        "chapters": [
            {
                "chapter_number": 1,
                "title": "Units and Measurement",
                "sections": [
                    {
                        "section_number": "1.1",
                        "title": "...",
                        "learning_objectives": [...],
                        "content": "...",
                        "glossary_terms": [{"term": "...", "definition": "..."}],
                        "key_equations": [...]
                    }
                ]
            }
        ]
    }
    """

    def __init__(self, subject: str, chunker: Optional[TextChunker] = None):
        self.subject = subject
        self.chunker = chunker or TextChunker()

    def process_book(self, book_data: Dict) -> Tuple[List[ConceptNode], List[CorpusChunk]]:
        """Process a complete OpenStax book."""
        all_concepts = []
        all_chunks = []
        global_chapter_order = 0

        for chapter in book_data.get("chapters", []):
            chapter_num = chapter.get("chapter_number", global_chapter_order)

            for section in chapter.get("sections", []):
                chapter_order = int(f"{chapter_num:03d}{section.get('section_number', '0').replace('.', ''):>03s}"[:6])

                # Extract concepts from glossary terms
                for term_entry in section.get("glossary_terms", []):
                    concept = self._make_concept(
                        term=term_entry.get("term", ""),
                        definition=term_entry.get("definition", ""),
                        chapter_order=chapter_order,
                        depth_level=self._estimate_depth(chapter_num, book_data)
                    )
                    if concept:
                        all_concepts.append(concept)

                # Extract concepts from learning objectives
                for i, objective in enumerate(section.get("learning_objectives", [])):
                    concept = self._make_concept(
                        term=objective.split(":")[0] if ":" in objective else objective[:60],
                        definition=objective,
                        chapter_order=chapter_order + i,
                        depth_level=self._estimate_depth(chapter_num, book_data)
                    )
                    if concept:
                        all_concepts.append(concept)

                # Chunk section content
                content = section.get("content", "")
                if content:
                    section_metadata = {
                        "chapter": chapter_num,
                        "section": section.get("section_number", ""),
                        "title": section.get("title", ""),
                        "source": "openstax",
                        "subject": self.subject,
                        "chapter_order": chapter_order,
                        "depth_level": self._estimate_depth(chapter_num, book_data)
                    }

                    raw_chunks = self.chunker.chunk(content, section_metadata)

                    for raw_chunk in raw_chunks:
                        # Find which concepts this chunk likely explains
                        chunk_concepts = self._identify_chunk_concepts(
                            raw_chunk["text"],
                            all_concepts
                        )

                        corpus_chunk = CorpusChunk(
                            chunk_id=generate_id(raw_chunk["text"], "openstax"),
                            text=raw_chunk["text"],
                            concept_ids=[c.concept_id for c in chunk_concepts],
                            prerequisite_concept_ids=[],  # Filled by graph builder
                            depth_level=section_metadata["depth_level"],
                            chapter_order=chapter_order,
                            subject=self.subject,
                            source="openstax",
                            metadata=section_metadata
                        )
                        all_chunks.append(corpus_chunk)

            global_chapter_order += 1

        logger.info(
            f"OpenStax [{self.subject}]: "
            f"{len(all_concepts)} concepts, {len(all_chunks)} chunks"
        )
        return all_concepts, all_chunks

    def _make_concept(
        self,
        term: str,
        definition: str,
        chapter_order: int,
        depth_level: int
    ) -> Optional[ConceptNode]:
        if len(term.strip()) < 2 or len(definition.strip()) < 10:
            return None
        return ConceptNode(
            concept_id=generate_id(term.lower(), "concept"),
            name=term.strip(),
            description=definition.strip(),
            source_chunk_ids=[],
            depth_level=depth_level,
            chapter_order=chapter_order,
            subject=self.subject,
            tags=["glossary"]
        )

    def _estimate_depth(self, chapter_num: int, book_data: Dict) -> int:
        """Estimate depth level (0/1/2) from chapter position in book."""
        total_chapters = len(book_data.get("chapters", []))
        if total_chapters == 0:
            return 0
        position = chapter_num / total_chapters
        if position < 0.33:
            return 0  # Introductory
        elif position < 0.67:
            return 1  # Intermediate
        else:
            return 2  # Advanced

    def _identify_chunk_concepts(
        self,
        chunk_text: str,
        concepts: List[ConceptNode],
        threshold: int = 2
    ) -> List[ConceptNode]:
        """Identify which concepts appear in a chunk (simple keyword matching)."""
        chunk_lower = chunk_text.lower()
        matching = []
        for concept in concepts:
            # Require both name and at least one key word from description
            name_words = concept.name.lower().split()
            if len(name_words) >= 1 and concept.name.lower() in chunk_lower:
                matching.append(concept)
        return matching[:10]  # Cap at 10 concepts per chunk


class CK12Processor:
    """
    Processes CK-12 Flexbooks which explicitly provide multiple depth levels
    for the same concept — the key advantage for depth modulation validation.

    CK-12 provides:
    - "Basic" level explanations (depth 0)
    - "Intermediate" level (depth 1)
    - "Advanced" level (depth 2)
    All for the same concept, enabling direct depth modulation evaluation.
    """

    def __init__(self, subject: str, chunker: Optional[TextChunker] = None):
        self.subject = subject
        self.chunker = chunker or TextChunker(chunk_size=200)

    def process_flexbook(
        self,
        flexbook_data: Dict
    ) -> Tuple[List[ConceptNode], List[CorpusChunk]]:
        """
        Process a CK-12 flexbook with explicit depth levels.

        Expected format:
        {
            "concept_name": "Newton's Second Law",
            "levels": {
                "basic": "Force equals mass times acceleration...",
                "intermediate": "F = ma where F is net force...",
                "advanced": "The second law states that dp/dt = F..."
            },
            "chapter_order": 5
        }
        """
        all_concepts = []
        all_chunks = []

        level_to_depth = {"basic": 0, "intermediate": 1, "advanced": 2}

        for item in flexbook_data.get("items", []):
            concept_name = item.get("concept_name", "")
            chapter_order = item.get("chapter_order", 0)

            # Create concept node (depth-agnostic)
            concept = ConceptNode(
                concept_id=generate_id(concept_name.lower(), "ck12"),
                name=concept_name,
                description=item.get("definition", concept_name),
                source_chunk_ids=[],
                depth_level=1,  # Canonical depth (actual chunks vary)
                chapter_order=chapter_order,
                subject=self.subject,
                tags=["ck12", "multi-depth"]
            )
            all_concepts.append(concept)

            # Create chunks at each depth level
            for level_name, level_text in item.get("levels", {}).items():
                depth = level_to_depth.get(level_name, 1)
                raw_chunks = self.chunker.chunk(
                    level_text,
                    {
                        "source": "ck12",
                        "concept": concept_name,
                        "depth_level": depth,
                        "level_name": level_name,
                        "chapter_order": chapter_order
                    }
                )

                for raw_chunk in raw_chunks:
                    corpus_chunk = CorpusChunk(
                        chunk_id=generate_id(
                            raw_chunk["text"] + level_name, "ck12"
                        ),
                        text=raw_chunk["text"],
                        concept_ids=[concept.concept_id],
                        prerequisite_concept_ids=[],
                        depth_level=depth,
                        chapter_order=chapter_order,
                        subject=self.subject,
                        source="ck12",
                        metadata=raw_chunk["metadata"]
                    )
                    all_chunks.append(corpus_chunk)
                    concept.source_chunk_ids.append(corpus_chunk.chunk_id)

        logger.info(
            f"CK-12 [{self.subject}]: "
            f"{len(all_concepts)} concepts, {len(all_chunks)} chunks"
        )
        return all_concepts, all_chunks


class AssistmentsProcessor:
    """
    Processes ASSISTments student interaction logs for KARMA training.

    ASSISTments CSV format:
    order_id, assignment_id, user_id, assistment_id, problem_id,
    original, correct, attempt_count, ms_first_response, tutor_mode,
    answer_type, sequence_id, student_class_id, position, type,
    base_sequence_id, skill_id, skill_name, teacher_id, school_id,
    hint_count, hint_total, overlap_time, template_id, answer_id,
    answer_text, first_action, bottom_hint, opportunity, opportunity_original
    """

    def __init__(self, concept_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            concept_mapping: Maps ASSISTments skill_name → concept_id in our graph
        """
        self.concept_mapping = concept_mapping or {}

    def process_csv(
        self,
        csv_path: str,
        max_students: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process ASSISTments interactions.

        Returns: Dict[student_id → List[interaction_dict]]
        """
        import csv

        student_interactions: Dict[str, List[Dict]] = {}
        n_processed = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                student_id = row.get("user_id", "")
                if not student_id:
                    continue

                if max_students and len(student_interactions) >= max_students:
                    if student_id not in student_interactions:
                        continue

                skill_name = row.get("skill_name", "")
                concept_id = self.concept_mapping.get(
                    skill_name, generate_id(skill_name, "skill")
                )

                correct = row.get("correct", "0") == "1"
                timestamp_ms = int(row.get("ms_first_response", 0))
                hint_count = int(row.get("hint_count", 0))

                # Response quality proxy: correct with no hints = high quality
                if correct and hint_count == 0:
                    quality = 1.0
                elif correct:
                    quality = max(0.5, 1.0 - 0.1 * hint_count)
                else:
                    quality = 0.2

                interaction = {
                    "interaction_id": f"assist_{row.get('order_id', n_processed)}",
                    "student_id": student_id,
                    "timestamp_ms": timestamp_ms,
                    "concept_id": concept_id,
                    "skill_name": skill_name,
                    "correct": correct,
                    "response_quality": quality,
                    "hint_count": hint_count,
                    "attempt_count": int(row.get("attempt_count", 1))
                }

                if student_id not in student_interactions:
                    student_interactions[student_id] = []
                student_interactions[student_id].append(interaction)
                n_processed += 1

        logger.info(
            f"ASSISTments: {len(student_interactions)} students, "
            f"{n_processed} total interactions"
        )
        return student_interactions

    def to_karma_interactions(
        self,
        student_interactions: List[Dict],
        base_timestamp: Optional[datetime] = None
    ) -> List[Dict]:
        """Convert ASSISTments interactions to KARMA Interaction-compatible dicts."""
        base = base_timestamp or datetime(2024, 1, 1)
        Interaction = _get_interaction_class()
        karma_interactions = []

        for item in sorted(student_interactions, key=lambda x: x["timestamp_ms"]):
            timestamp = datetime.fromtimestamp(
                base.timestamp() + item["timestamp_ms"] / 1000.0
            )

            karma_interactions.append(Interaction(
                interaction_id=item["interaction_id"],
                timestamp=timestamp,
                query=f"[Assessment: {item['skill_name']}]",
                concept_ids=[item["concept_id"]],
                correct=item["correct"],
                response_quality=item["response_quality"],
                mrl_divergence=0.0  # Not available in ASSISTments
            ))

        return karma_interactions


class LectureBankProcessor:
    """
    Processes LectureBank prerequisite annotations.

    LectureBank format (Kann et al., 2019):
    concepts.txt: one concept per line
    prereqs.txt: "concept_A\tconcept_B\tlabel" (1=prereq, 0=not)
    """

    def process(
        self,
        concepts_path: str,
        prereqs_path: str,
        concept_id_mapping: Optional[Dict[str, str]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Process LectureBank annotations into prerequisite edge list.

        Returns: List of (prereq_concept_id, dependent_concept_id, confidence)
        """
        mapping = concept_id_mapping or {}

        # Load concepts
        concepts = {}
        with open(concepts_path) as f:
            for line in f:
                name = line.strip()
                if name:
                    cid = mapping.get(name, generate_id(name, "lb"))
                    concepts[name] = cid

        # Load prerequisite annotations
        prereq_edges = []
        with open(prereqs_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    concept_a, concept_b, label = parts[0], parts[1], parts[2]
                    if label == "1" and concept_a in concepts and concept_b in concepts:
                        prereq_edges.append((
                            concepts[concept_a],
                            concepts[concept_b],
                            0.9  # Human-annotated = high confidence
                        ))

        logger.info(f"LectureBank: {len(prereq_edges)} prerequisite edges")
        return prereq_edges


class MOOCCubeProcessor:
    """
    Processes MOOCCube dataset (Yu et al., 2020, KDD).

    MOOCCube provides both:
    - Concept prerequisite graphs (for E_P construction)
    - Student behavior logs (for KARMA training)

    This makes it the most valuable single dataset for PLEDGE-KARMA.
    """

    def process_concept_relations(
        self,
        relations_path: str
    ) -> List[Tuple[str, str, float]]:
        """
        Process MOOCCube concept prerequisite relations.

        Relations file format (JSON Lines):
        {"source": "concept_A", "target": "concept_B", "score": 0.85}
        """
        prereq_edges = []
        with open(relations_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rel = json.loads(line)
                    source_id = generate_id(rel["source"], "mc")
                    target_id = generate_id(rel["target"], "mc")
                    score = float(rel.get("score", 0.7))
                    prereq_edges.append((source_id, target_id, score))
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info(f"MOOCCube: {len(prereq_edges)} prerequisite relations")
        return prereq_edges

    def process_student_logs(
        self,
        logs_path: str,
        max_students: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process MOOCCube student behavior logs.

        Log format (JSON Lines):
        {"user_id": "u123", "concept_id": "c456", "video_id": "v789",
         "timestamp": 1234567890, "watch_ratio": 0.85}
        """
        student_logs: Dict[str, List[Dict]] = {}

        with open(logs_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    log = json.loads(line)
                    uid = log["user_id"]

                    if max_students and len(student_logs) >= max_students:
                        if uid not in student_logs:
                            continue

                    # Watch ratio as proxy for engagement/correctness
                    watch_ratio = float(log.get("watch_ratio", 0.5))
                    quality = watch_ratio  # Higher watch = more engagement

                    interaction = {
                        "student_id": uid,
                        "concept_id": generate_id(log["concept_id"], "mc"),
                        "timestamp": log.get("timestamp", 0),
                        "response_quality": quality,
                        "correct": watch_ratio > 0.8,  # Heuristic
                    }

                    if uid not in student_logs:
                        student_logs[uid] = []
                    student_logs[uid].append(interaction)

                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info(
            f"MOOCCube: {len(student_logs)} students from behavior logs"
        )
        return student_logs